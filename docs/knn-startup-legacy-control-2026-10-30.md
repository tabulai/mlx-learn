# Predeclared Digits, Breast Cancer and offset legacy-order control

All **60 operation cells**, both automatic replicas and all five workers are retained. Accuracy, runtime identity and numerical-oracle validation pass. The two isolated >5% full-control flags were covered by these predeclared panels; the panel-specific crosschecks below retain both the full result and the repeat, including smaller or inconsistent slowdowns.

This is the `legacy-control` portion of the [saved predeclaration](../benchmarks/results/knn-startup-final2-legacy-predeclaration-2026-10-30.json) (SHA-256 `882bbe578f6a0e94c3e172f4d2bfcc7d801a63ac7a03bc985e9fb74f5ab3f131`). The original Digits/Breast/offset panel was retained, and all six Gaussian 128D seeds plus all three clustered 128D seeds were added before the final preservation results. The exact requested inventories are verified, with no seed or operation removed.

Each operation uses 31 serial interleaved rounds across current auto, identical current auto A/A, current forced GPU, immediate prior auto (`knn-latency-final-2026-10-30` plus its exact archived native runtime), and historical forced GPU (`knn-small-final-2026-10-30`). This panel uses the legacy forward/reverse ordering and 1 second of continuous warmup. The full 48-fixture control uses balanced Williams ordering and 0.25 seconds of warmup. Absolute milliseconds across these runs are not directly comparable; ratios compare implementations within their own run. Neither run replaces the other, and no faster window or run is selected.

M/P means ratio of medians / median paired ratio. Ratios are comparator/current; below 1× means the automatic replica is slower and is bold. A >5% runtime slowdown means ratio <1/1.05 for BOTH statistics and BOTH independent automatic replicas. This recurrence screen is descriptive, not a confidence interval or significance test.

## Full-control exceptions and repeat crosscheck

| Comparator | Flags in full 222 cells | Covered here | Recurring | Not recurring | New flags |
| --- | --- | --- | --- | --- | --- |
| Immediate prior auto | 1 | 1 | 0 | 1 | 0 |
| Current forced GPU | 1 | 0 | 0 | 0 | 0 |
| Historical forced GPU | 0 | 0 | 0 | 0 | 0 |

| Comparator | Classification | Family | Seed | Operation | Full current M/P | Full A/A M/P | Repeat current M/P | Repeat A/A M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Immediate prior auto | not recurring | Digits | 151 | regress_distance | **0.9003×** / **0.8917×** | **0.8793×** / **0.9040×** | 1.1086× / 1.0786× | 1.0847× / 1.0860× |

## Every slower observation in either statistic

| Comparator | Replica | Family | Seed | Operation | M/P |
| --- | --- | --- | --- | --- | --- |
| Historical forced GPU | Current auto | Digits | 252 | regress_distance | **0.9642×** / **0.9794×** |
| Historical forced GPU | Current auto A/A | Digits | 252 | regress_distance | **0.9842×** / **0.9977×** |
| Immediate prior auto | Current auto | Offset +1e4, 4096×128 | 151 | kneighbors | **0.9815×** / 1.0138× |
| Immediate prior auto | Current auto A/A | Offset +1e4, 4096×128 | 151 | kneighbors | **0.9566×** / **0.9840×** |
| Current forced GPU | Current auto | Offset +1e4, 4096×128 | 151 | kneighbors | **0.9830×** / 1.0153× |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 151 | kneighbors | **0.9581×** / **0.9936×** |
| Historical forced GPU | Current auto | Offset +1e4, 4096×128 | 151 | kneighbors | **0.9976×** / 1.0032× |
| Historical forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 151 | kneighbors | **0.9724×** / 1.0035× |
| Immediate prior auto | Current auto | Offset +1e4, 4096×128 | 251 | kneighbors | **0.9832×** / 1.0303× |
| Immediate prior auto | Current auto A/A | Offset +1e4, 4096×128 | 251 | kneighbors | **0.9519×** / 1.0071× |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 251 | kneighbors | **0.9847×** / **0.9469×** |
| Historical forced GPU | Current auto | Offset +1e4, 4096×128 | 251 | kneighbors | **0.9872×** / **0.9974×** |
| Historical forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 251 | kneighbors | **0.9558×** / 1.0392× |
| Immediate prior auto | Current auto | Offset +1e4, 4096×128 | 251 | fit_query | **0.9981×** / 1.0402× |
| Current forced GPU | Current auto | Offset +1e4, 4096×128 | 251 | fit_query | **0.9186×** / **0.9971×** |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 251 | fit_query | **0.9479×** / **0.9965×** |
| Immediate prior auto | Current auto | Offset +1e4, 4096×128 | 252 | kneighbors | 1.0139× / **0.9893×** |
| Immediate prior auto | Current auto A/A | Offset +1e4, 4096×128 | 252 | kneighbors | 1.0172× / **0.9863×** |
| Current forced GPU | Current auto | Offset +1e4, 4096×128 | 252 | kneighbors | **0.9544×** / **0.9582×** |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 252 | kneighbors | **0.9575×** / **0.9707×** |
| Historical forced GPU | Current auto | Offset +1e4, 4096×128 | 252 | kneighbors | 1.0125× / **0.9759×** |
| Current forced GPU | Current auto | Offset +1e4, 4096×128 | 252 | fit_query | **0.9506×** / **0.9906×** |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 252 | fit_query | **0.9443×** / **0.9900×** |
| Historical forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 252 | fit_query | **0.9987×** / 1.0784× |

## Every repeat timing cell

All times are median milliseconds from this repeat only.

| Family | Seed | Operation | Current auto ms | Current auto A/A ms | Current forced GPU ms | Immediate prior auto ms | Historical forced GPU ms | A/A over auto M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 0.474708 | 0.461292 | 0.885000 | 0.583750 | 0.612459 | **0.9717×** / **0.9708×** |
| Digits | 151 | kneighbors | 0.293959 | 0.376750 | 0.930000 | 0.428708 | 0.874375 | 1.2816× / 1.1846× |
| Digits | 151 | indices | 0.375667 | 0.309417 | 0.477334 | 0.431250 | 0.963417 | **0.8236×** / **0.7999×** |
| Digits | 151 | predict_uniform | 0.375459 | 0.376042 | 0.428958 | 0.445500 | 0.450792 | 1.0016× / **0.9517×** |
| Digits | 151 | proba_uniform | 0.400917 | 0.393584 | 0.998875 | 0.404208 | 0.955167 | **0.9817×** / 1.0082× |
| Digits | 151 | regress_uniform | 0.378541 | 0.387667 | 0.524583 | 0.427791 | 0.400458 | 1.0241× / 1.0432× |
| Digits | 151 | predict_distance | 0.334875 | 0.402625 | 0.523667 | 0.465250 | 0.641167 | 1.2023× / 1.2048× |
| Digits | 151 | proba_distance | 0.418958 | 0.358333 | 0.449167 | 0.484708 | 0.959667 | **0.8553×** / **0.8579×** |
| Digits | 151 | regress_distance | 0.416750 | 0.425916 | 0.447000 | 0.462000 | 0.456542 | 1.0220× / 1.0050× |
| Digits | 251 | fit_query | 0.444416 | 0.498500 | 1.052083 | 0.596667 | 1.069959 | 1.1217× / 1.1100× |
| Digits | 251 | kneighbors | 0.393333 | 0.333375 | 0.976584 | 0.453083 | 0.889750 | **0.8476×** / **0.8520×** |
| Digits | 251 | indices | 0.384417 | 0.371291 | 0.443250 | 0.437625 | 0.937584 | **0.9659×** / **0.9640×** |
| Digits | 251 | predict_uniform | 0.421916 | 0.416625 | 0.858917 | 0.492584 | 0.476917 | **0.9875×** / **0.9867×** |
| Digits | 251 | proba_uniform | 0.384916 | 0.406250 | 0.529000 | 0.467958 | 0.459750 | 1.0554× / **0.9815×** |
| Digits | 251 | regress_uniform | 0.339416 | 0.403166 | 0.924625 | 0.461084 | 0.928792 | 1.1878× / 1.1594× |
| Digits | 251 | predict_distance | 0.403541 | 0.367791 | 0.691334 | 0.476042 | 0.974417 | **0.9114×** / **0.9111×** |
| Digits | 251 | proba_distance | 0.433250 | 0.428375 | 0.463750 | 0.471166 | 0.478333 | **0.9887×** / **0.9705×** |
| Digits | 251 | regress_distance | 0.433542 | 0.426666 | 0.992333 | 0.434209 | 0.677791 | **0.9841×** / 1.0254× |
| Digits | 252 | fit_query | 0.511542 | 0.462541 | 0.765375 | 0.621084 | 0.921333 | **0.9042×** / **0.9002×** |
| Digits | 252 | kneighbors | 0.388417 | 0.383250 | 0.416625 | 0.455958 | 0.880417 | **0.9867×** / **0.9835×** |
| Digits | 252 | indices | 0.386292 | 0.388542 | 0.861333 | 0.413166 | 0.913708 | 1.0058× / 1.0088× |
| Digits | 252 | predict_uniform | 0.344709 | 0.384000 | 0.911000 | 0.458458 | 0.456125 | 1.1140× / 1.1031× |
| Digits | 252 | proba_uniform | 0.334042 | 0.409958 | 1.063375 | 0.481167 | 1.303042 | 1.2273× / 1.1680× |
| Digits | 252 | regress_uniform | 0.381000 | 0.327250 | 0.907791 | 0.465542 | 0.929709 | **0.8589×** / **0.9030×** |
| Digits | 252 | predict_distance | 0.420792 | 0.390375 | 0.446250 | 0.481584 | 0.467375 | **0.9277×** / **0.9610×** |
| Digits | 252 | proba_distance | 0.431708 | 0.421666 | 0.888791 | 0.452791 | 0.906375 | **0.9767×** / 1.0005× |
| Digits | 252 | regress_distance | 0.420250 | 0.411708 | 0.850125 | 0.448167 | 0.405208 | **0.9797×** / **0.9924×** |
| Breast Cancer | 151 | fit_query | 0.222333 | 0.227583 | 0.287292 | 0.299041 | 0.327167 | 1.0236× / 1.0174× |
| Breast Cancer | 151 | kneighbors | 0.156250 | 0.150584 | 0.264500 | 0.179458 | 0.262458 | **0.9637×** / **0.9961×** |
| Breast Cancer | 151 | indices | 0.162250 | 0.155333 | 0.235708 | 0.199042 | 0.218958 | **0.9574×** / 1.0705× |
| Breast Cancer | 151 | predict_uniform | 0.143000 | 0.153333 | 0.303500 | 0.195541 | 0.293250 | 1.0723× / 1.1418× |
| Breast Cancer | 151 | proba_uniform | 0.165375 | 0.143083 | 0.309375 | 0.208208 | 0.322500 | **0.8652×** / **0.9127×** |
| Breast Cancer | 151 | regress_uniform | 0.162625 | 0.176292 | 0.239709 | 0.202875 | 0.244500 | 1.0840× / 1.0546× |
| Breast Cancer | 151 | predict_distance | 0.180209 | 0.183542 | 0.323417 | 0.205875 | 0.278167 | 1.0185× / 1.0251× |
| Breast Cancer | 151 | proba_distance | 0.170208 | 0.207750 | 0.271583 | 0.224375 | 0.248542 | 1.2206× / 1.1406× |
| Breast Cancer | 151 | regress_distance | 0.162584 | 0.173333 | 0.297708 | 0.214000 | 0.247708 | 1.0661× / 1.1092× |
| Breast Cancer | 251 | fit_query | 0.233042 | 0.226250 | 0.343083 | 0.287333 | 0.388792 | **0.9709×** / **0.9783×** |
| Breast Cancer | 251 | kneighbors | 0.154125 | 0.161375 | 0.247250 | 0.195000 | 0.225584 | 1.0470× / 1.0306× |
| Breast Cancer | 251 | indices | 0.131250 | 0.154041 | 0.262916 | 0.195333 | 0.258959 | 1.1736× / 1.1825× |
| Breast Cancer | 251 | predict_uniform | 0.157500 | 0.136708 | 0.287375 | 0.204375 | 0.324375 | **0.8680×** / **0.8710×** |
| Breast Cancer | 251 | proba_uniform | 0.176917 | 0.157458 | 0.252625 | 0.199750 | 0.261958 | **0.8900×** / **0.9304×** |
| Breast Cancer | 251 | regress_uniform | 0.175042 | 0.173250 | 0.255250 | 0.183375 | 0.262458 | **0.9898×** / **0.9706×** |
| Breast Cancer | 251 | predict_distance | 0.182125 | 0.191916 | 0.276750 | 0.221167 | 0.251875 | 1.0538× / 1.0189× |
| Breast Cancer | 251 | proba_distance | 0.155250 | 0.175833 | 0.413500 | 0.216125 | 0.293000 | 1.1326× / 1.1274× |
| Breast Cancer | 251 | regress_distance | 0.179417 | 0.152791 | 0.267334 | 0.213125 | 0.502958 | **0.8516×** / **0.8284×** |
| Breast Cancer | 252 | fit_query | 0.230084 | 0.225000 | 0.351500 | 0.294209 | 0.322250 | **0.9779×** / **0.9931×** |
| Breast Cancer | 252 | kneighbors | 0.140417 | 0.155042 | 0.242500 | 0.189584 | 0.246667 | 1.1042× / 1.1218× |
| Breast Cancer | 252 | indices | 0.149792 | 0.126917 | 0.239167 | 0.196875 | 0.257792 | **0.8473×** / **0.8786×** |
| Breast Cancer | 252 | predict_uniform | 0.175625 | 0.170958 | 0.248958 | 0.214416 | 0.258208 | **0.9734×** / **0.9933×** |
| Breast Cancer | 252 | proba_uniform | 0.180792 | 0.162125 | 0.303917 | 0.185833 | 0.305750 | **0.8967×** / **0.9517×** |
| Breast Cancer | 252 | regress_uniform | 0.151333 | 0.167334 | 0.249167 | 0.207208 | 0.229292 | 1.1057× / 1.0479× |
| Breast Cancer | 252 | predict_distance | 0.157541 | 0.203750 | 0.286959 | 0.225875 | 0.296625 | 1.2933× / 1.1948× |
| Breast Cancer | 252 | proba_distance | 0.186458 | 0.162000 | 0.299625 | 0.230042 | 0.363042 | **0.8688×** / **0.8370×** |
| Breast Cancer | 252 | regress_distance | 0.187834 | 0.183375 | 0.259250 | 0.221250 | 0.263208 | **0.9763×** / 1.0135× |
| Offset +1e4, 4096×128 | 151 | kneighbors | 0.420167 | 0.431083 | 0.413041 | 0.412375 | 0.419167 | 1.0260× / 1.0658× |
| Offset +1e4, 4096×128 | 151 | fit_query | 0.644416 | 0.644000 | 0.689209 | 0.743125 | 0.773334 | **0.9994×** / **0.9712×** |
| Offset +1e4, 4096×128 | 251 | kneighbors | 0.446042 | 0.460708 | 0.453666 | 0.438542 | 0.440334 | 1.0329× / **0.9951×** |
| Offset +1e4, 4096×128 | 251 | fit_query | 0.706166 | 0.684291 | 0.648667 | 0.704792 | 0.740542 | **0.9690×** / **0.9715×** |
| Offset +1e4, 4096×128 | 252 | kneighbors | 0.424625 | 0.423250 | 0.405250 | 0.430542 | 0.429916 | **0.9968×** / **0.9741×** |
| Offset +1e4, 4096×128 | 252 | fit_query | 0.661041 | 0.665416 | 0.628375 | 0.670958 | 0.664541 | 1.0066× / **0.9706×** |

## Every cell against Immediate prior auto

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.2297× | 1.2082× | 1.2655× | 1.2143× |
| Digits | 151 | kneighbors | 1.4584× | 1.4342× | 1.1379× | 1.1712× |
| Digits | 151 | indices | 1.1480× | 1.1488× | 1.3937× | 1.4666× |
| Digits | 151 | predict_uniform | 1.1865× | 1.1465× | 1.1847× | 1.2185× |
| Digits | 151 | proba_uniform | 1.0082× | 1.0613× | 1.0270× | 1.0498× |
| Digits | 151 | regress_uniform | 1.1301× | 1.0794× | 1.1035× | 1.0887× |
| Digits | 151 | predict_distance | 1.3893× | 1.3280× | 1.1555× | 1.1161× |
| Digits | 151 | proba_distance | 1.1569× | 1.1520× | 1.3527× | 1.3686× |
| Digits | 151 | regress_distance | 1.1086× | 1.0786× | 1.0847× | 1.0860× |
| Digits | 251 | fit_query | 1.3426× | 1.3334× | 1.1969× | 1.2269× |
| Digits | 251 | kneighbors | 1.1519× | 1.1708× | 1.3591× | 1.3205× |
| Digits | 251 | indices | 1.1384× | 1.1614× | 1.1787× | 1.1916× |
| Digits | 251 | predict_uniform | 1.1675× | 1.1361× | 1.1823× | 1.1264× |
| Digits | 251 | proba_uniform | 1.2157× | 1.1580× | 1.1519× | 1.1553× |
| Digits | 251 | regress_uniform | 1.3585× | 1.2973× | 1.1437× | 1.1180× |
| Digits | 251 | predict_distance | 1.1797× | 1.1884× | 1.2943× | 1.3062× |
| Digits | 251 | proba_distance | 1.0875× | 1.0695× | 1.0999× | 1.1430× |
| Digits | 251 | regress_distance | 1.0015× | 1.0147× | 1.0177× | 1.0066× |
| Digits | 252 | fit_query | 1.2141× | 1.2157× | 1.3428× | 1.3645× |
| Digits | 252 | kneighbors | 1.1739× | 1.1607× | 1.1897× | 1.1947× |
| Digits | 252 | indices | 1.0696× | 1.1332× | 1.0634× | 1.0694× |
| Digits | 252 | predict_uniform | 1.3300× | 1.3262× | 1.1939× | 1.1949× |
| Digits | 252 | proba_uniform | 1.4404× | 1.3703× | 1.1737× | 1.1606× |
| Digits | 252 | regress_uniform | 1.2219× | 1.1608× | 1.4226× | 1.3408× |
| Digits | 252 | predict_distance | 1.1445× | 1.1208× | 1.2336× | 1.1374× |
| Digits | 252 | proba_distance | 1.0488× | 1.0042× | 1.0738× | 1.0638× |
| Digits | 252 | regress_distance | 1.0664× | 1.0831× | 1.0886× | 1.0923× |
| Breast Cancer | 151 | fit_query | 1.3450× | 1.3360× | 1.3140× | 1.3770× |
| Breast Cancer | 151 | kneighbors | 1.1485× | 1.1036× | 1.1917× | 1.1937× |
| Breast Cancer | 151 | indices | 1.2268× | 1.2743× | 1.2814× | 1.1945× |
| Breast Cancer | 151 | predict_uniform | 1.3674× | 1.4395× | 1.2753× | 1.2219× |
| Breast Cancer | 151 | proba_uniform | 1.2590× | 1.2629× | 1.4552× | 1.4556× |
| Breast Cancer | 151 | regress_uniform | 1.2475× | 1.2457× | 1.1508× | 1.1691× |
| Breast Cancer | 151 | predict_distance | 1.1424× | 1.0831× | 1.1217× | 1.0988× |
| Breast Cancer | 151 | proba_distance | 1.3182× | 1.3355× | 1.0800× | 1.1399× |
| Breast Cancer | 151 | regress_distance | 1.3162× | 1.3259× | 1.2346× | 1.1941× |
| Breast Cancer | 251 | fit_query | 1.2330× | 1.2154× | 1.2700× | 1.2931× |
| Breast Cancer | 251 | kneighbors | 1.2652× | 1.2363× | 1.2084× | 1.2132× |
| Breast Cancer | 251 | indices | 1.4883× | 1.4514× | 1.2681× | 1.1846× |
| Breast Cancer | 251 | predict_uniform | 1.2976× | 1.2734× | 1.4950× | 1.4936× |
| Breast Cancer | 251 | proba_uniform | 1.1291× | 1.1713× | 1.2686× | 1.2910× |
| Breast Cancer | 251 | regress_uniform | 1.0476× | 1.0289× | 1.0584× | 1.0367× |
| Breast Cancer | 251 | predict_distance | 1.2144× | 1.1423× | 1.1524× | 1.1102× |
| Breast Cancer | 251 | proba_distance | 1.3921× | 1.3842× | 1.2291× | 1.2259× |
| Breast Cancer | 251 | regress_distance | 1.1879× | 1.1760× | 1.3949× | 1.4340× |
| Breast Cancer | 252 | fit_query | 1.2787× | 1.2847× | 1.3076× | 1.3450× |
| Breast Cancer | 252 | kneighbors | 1.3502× | 1.3863× | 1.2228× | 1.2110× |
| Breast Cancer | 252 | indices | 1.3143× | 1.3186× | 1.5512× | 1.5523× |
| Breast Cancer | 252 | predict_uniform | 1.2209× | 1.1960× | 1.2542× | 1.2820× |
| Breast Cancer | 252 | proba_uniform | 1.0279× | 1.0292× | 1.1462× | 1.1063× |
| Breast Cancer | 252 | regress_uniform | 1.3692× | 1.3153× | 1.2383× | 1.2435× |
| Breast Cancer | 252 | predict_distance | 1.4338× | 1.3673× | 1.1086× | 1.1246× |
| Breast Cancer | 252 | proba_distance | 1.2337× | 1.1928× | 1.4200× | 1.4853× |
| Breast Cancer | 252 | regress_distance | 1.1779× | 1.1862× | 1.2065× | 1.2164× |
| Offset +1e4, 4096×128 | 151 | kneighbors | **0.9815×** | 1.0138× | **0.9566×** | **0.9840×** |
| Offset +1e4, 4096×128 | 151 | fit_query | 1.1532× | 1.0767× | 1.1539× | 1.0916× |
| Offset +1e4, 4096×128 | 251 | kneighbors | **0.9832×** | 1.0303× | **0.9519×** | 1.0071× |
| Offset +1e4, 4096×128 | 251 | fit_query | **0.9981×** | 1.0402× | 1.0300× | 1.0299× |
| Offset +1e4, 4096×128 | 252 | kneighbors | 1.0139× | **0.9893×** | 1.0172× | **0.9863×** |
| Offset +1e4, 4096×128 | 252 | fit_query | 1.0150× | 1.0378× | 1.0083× | 1.0154× |

## Every cell against Current forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.8643× | 1.8977× | 1.9185× | 1.7503× |
| Digits | 151 | kneighbors | 3.1637× | 2.7929× | 2.4685× | 2.5060× |
| Digits | 151 | indices | 1.2706× | 1.4889× | 1.5427× | 1.5840× |
| Digits | 151 | predict_uniform | 1.1425× | 1.1192× | 1.1407× | 1.2390× |
| Digits | 151 | proba_uniform | 2.4915× | 2.7871× | 2.5379× | 2.6945× |
| Digits | 151 | regress_uniform | 1.3858× | 1.3419× | 1.3532× | 1.2712× |
| Digits | 151 | predict_distance | 1.5638× | 1.6913× | 1.3006× | 1.2453× |
| Digits | 151 | proba_distance | 1.0721× | 1.2051× | 1.2535× | 1.3527× |
| Digits | 151 | regress_distance | 1.0726× | 1.1641× | 1.0495× | 1.2055× |
| Digits | 251 | fit_query | 2.3673× | 2.3408× | 2.1105× | 2.1572× |
| Digits | 251 | kneighbors | 2.4828× | 2.5426× | 2.9294× | 2.8236× |
| Digits | 251 | indices | 1.1530× | 1.2226× | 1.1938× | 1.3088× |
| Digits | 251 | predict_uniform | 2.0358× | 2.0138× | 2.0616× | 1.9231× |
| Digits | 251 | proba_uniform | 1.3743× | 1.2708× | 1.3022× | 1.4482× |
| Digits | 251 | regress_uniform | 2.7242× | 2.5565× | 2.2934× | 2.2498× |
| Digits | 251 | predict_distance | 1.7132× | 1.7933× | 1.8797× | 2.0102× |
| Digits | 251 | proba_distance | 1.0704× | 1.0987× | 1.0826× | 1.1276× |
| Digits | 251 | regress_distance | 2.2889× | 2.2949× | 2.3258× | 2.3067× |
| Digits | 252 | fit_query | 1.4962× | 1.5226× | 1.6547× | 1.6914× |
| Digits | 252 | kneighbors | 1.0726× | 1.1895× | 1.0871× | 1.2346× |
| Digits | 252 | indices | 2.2297× | 2.1507× | 2.2168× | 2.4040× |
| Digits | 252 | predict_uniform | 2.6428× | 2.6374× | 2.3724× | 2.4679× |
| Digits | 252 | proba_uniform | 3.1834× | 3.2522× | 2.5939× | 2.7148× |
| Digits | 252 | regress_uniform | 2.3827× | 2.0592× | 2.7740× | 2.4536× |
| Digits | 252 | predict_distance | 1.0605× | 1.1389× | 1.1431× | 1.1501× |
| Digits | 252 | proba_distance | 2.0588× | 1.9113× | 2.1078× | 1.9067× |
| Digits | 252 | regress_distance | 2.0229× | 2.0411× | 2.0649× | 2.0977× |
| Breast Cancer | 151 | fit_query | 1.2922× | 1.2566× | 1.2624× | 1.3070× |
| Breast Cancer | 151 | kneighbors | 1.6928× | 1.7452× | 1.7565× | 1.6992× |
| Breast Cancer | 151 | indices | 1.4527× | 1.6651× | 1.5174× | 1.5693× |
| Breast Cancer | 151 | predict_uniform | 2.1224× | 2.1398× | 1.9794× | 1.9677× |
| Breast Cancer | 151 | proba_uniform | 1.8707× | 1.8681× | 2.1622× | 2.2123× |
| Breast Cancer | 151 | regress_uniform | 1.4740× | 1.5383× | 1.3597× | 1.3988× |
| Breast Cancer | 151 | predict_distance | 1.7947× | 1.7070× | 1.7621× | 1.7232× |
| Breast Cancer | 151 | proba_distance | 1.5956× | 1.6169× | 1.3073× | 1.4425× |
| Breast Cancer | 151 | regress_distance | 1.8311× | 1.7898× | 1.7175× | 1.6155× |
| Breast Cancer | 251 | fit_query | 1.4722× | 1.4972× | 1.5164× | 1.5907× |
| Breast Cancer | 251 | kneighbors | 1.6042× | 1.7160× | 1.5321× | 1.6374× |
| Breast Cancer | 251 | indices | 2.0032× | 1.9723× | 1.7068× | 1.6813× |
| Breast Cancer | 251 | predict_uniform | 1.8246× | 1.9622× | 2.1021× | 2.0241× |
| Breast Cancer | 251 | proba_uniform | 1.4279× | 1.5389× | 1.6044× | 1.6076× |
| Breast Cancer | 251 | regress_uniform | 1.4582× | 1.5428× | 1.4733× | 1.5906× |
| Breast Cancer | 251 | predict_distance | 1.5196× | 1.5623× | 1.4420× | 1.4857× |
| Breast Cancer | 251 | proba_distance | 2.6634× | 2.6549× | 2.3517× | 2.2560× |
| Breast Cancer | 251 | regress_distance | 1.4900× | 1.5427× | 1.7497× | 1.7949× |
| Breast Cancer | 252 | fit_query | 1.5277× | 1.5288× | 1.5622× | 1.6965× |
| Breast Cancer | 252 | kneighbors | 1.7270× | 1.7804× | 1.5641× | 1.6722× |
| Breast Cancer | 252 | indices | 1.5967× | 1.6382× | 1.8844× | 1.9196× |
| Breast Cancer | 252 | predict_uniform | 1.4176× | 1.4234× | 1.4563× | 1.4679× |
| Breast Cancer | 252 | proba_uniform | 1.6810× | 1.7974× | 1.8746× | 1.8840× |
| Breast Cancer | 252 | regress_uniform | 1.6465× | 1.6191× | 1.4890× | 1.5133× |
| Breast Cancer | 252 | predict_distance | 1.8215× | 1.8333× | 1.4084× | 1.6224× |
| Breast Cancer | 252 | proba_distance | 1.6069× | 1.6304× | 1.8495× | 1.9152× |
| Breast Cancer | 252 | regress_distance | 1.3802× | 1.4199× | 1.4138× | 1.4839× |
| Offset +1e4, 4096×128 | 151 | kneighbors | **0.9830×** | 1.0153× | **0.9581×** | **0.9936×** |
| Offset +1e4, 4096×128 | 151 | fit_query | 1.0695× | 1.0678× | 1.0702× | 1.1231× |
| Offset +1e4, 4096×128 | 251 | kneighbors | 1.0171× | 1.0072× | **0.9847×** | **0.9469×** |
| Offset +1e4, 4096×128 | 251 | fit_query | **0.9186×** | **0.9971×** | **0.9479×** | **0.9965×** |
| Offset +1e4, 4096×128 | 252 | kneighbors | **0.9544×** | **0.9582×** | **0.9575×** | **0.9707×** |
| Offset +1e4, 4096×128 | 252 | fit_query | **0.9506×** | **0.9906×** | **0.9443×** | **0.9900×** |

## Every cell against Historical forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.2902× | 1.2963× | 1.3277× | 1.3105× |
| Digits | 151 | kneighbors | 2.9745× | 2.4182× | 2.3208× | 2.2056× |
| Digits | 151 | indices | 2.5646× | 2.7120× | 3.1137× | 3.0823× |
| Digits | 151 | predict_uniform | 1.2006× | 1.1763× | 1.1988× | 1.2715× |
| Digits | 151 | proba_uniform | 2.3825× | 2.4944× | 2.4268× | 2.6554× |
| Digits | 151 | regress_uniform | 1.0579× | 1.0251× | 1.0330× | 1.0298× |
| Digits | 151 | predict_distance | 1.9146× | 2.1509× | 1.5925× | 1.5341× |
| Digits | 151 | proba_distance | 2.2906× | 2.3707× | 2.6781× | 2.5703× |
| Digits | 151 | regress_distance | 1.0955× | 1.1276× | 1.0719× | 1.1887× |
| Digits | 251 | fit_query | 2.4076× | 2.3318× | 2.1464× | 2.1195× |
| Digits | 251 | kneighbors | 2.2621× | 2.4327× | 2.6689× | 2.5024× |
| Digits | 251 | indices | 2.4390× | 2.3202× | 2.5252× | 2.3473× |
| Digits | 251 | predict_uniform | 1.1304× | 1.3608× | 1.1447× | 1.3462× |
| Digits | 251 | proba_uniform | 1.1944× | 1.2003× | 1.1317× | 1.1102× |
| Digits | 251 | regress_uniform | 2.7364× | 2.4654× | 2.3037× | 2.1696× |
| Digits | 251 | predict_distance | 2.4147× | 2.3789× | 2.6494× | 2.4737× |
| Digits | 251 | proba_distance | 1.1041× | 1.2055× | 1.1166× | 1.2590× |
| Digits | 251 | regress_distance | 1.5634× | 1.6712× | 1.5886× | 1.4902× |
| Digits | 252 | fit_query | 1.8011× | 1.7220× | 1.9919× | 2.0613× |
| Digits | 252 | kneighbors | 2.2667× | 2.3340× | 2.2972× | 2.2044× |
| Digits | 252 | indices | 2.3653× | 2.4938× | 2.3516× | 2.3628× |
| Digits | 252 | predict_uniform | 1.3232× | 1.3512× | 1.1878× | 1.1903× |
| Digits | 252 | proba_uniform | 3.9008× | 3.2214× | 3.1785× | 2.7279× |
| Digits | 252 | regress_uniform | 2.4402× | 2.5046× | 2.8410× | 2.7535× |
| Digits | 252 | predict_distance | 1.1107× | 1.1553× | 1.1972× | 1.1925× |
| Digits | 252 | proba_distance | 2.0995× | 2.1860× | 2.1495× | 2.2183× |
| Digits | 252 | regress_distance | **0.9642×** | **0.9794×** | **0.9842×** | **0.9977×** |
| Breast Cancer | 151 | fit_query | 1.4715× | 1.5054× | 1.4376× | 1.5224× |
| Breast Cancer | 151 | kneighbors | 1.6797× | 1.6961× | 1.7429× | 1.7371× |
| Breast Cancer | 151 | indices | 1.3495× | 1.4617× | 1.4096× | 1.3264× |
| Breast Cancer | 151 | predict_uniform | 2.0507× | 2.0137× | 1.9125× | 1.7935× |
| Breast Cancer | 151 | proba_uniform | 1.9501× | 2.0855× | 2.2539× | 2.2546× |
| Breast Cancer | 151 | regress_uniform | 1.5035× | 1.5510× | 1.3869× | 1.4721× |
| Breast Cancer | 151 | predict_distance | 1.5436× | 1.6451× | 1.5155× | 1.5793× |
| Breast Cancer | 151 | proba_distance | 1.4602× | 1.2938× | 1.1964× | 1.2521× |
| Breast Cancer | 151 | regress_distance | 1.5236× | 1.5572× | 1.4291× | 1.3960× |
| Breast Cancer | 251 | fit_query | 1.6683× | 1.5835× | 1.7184× | 1.7463× |
| Breast Cancer | 251 | kneighbors | 1.4636× | 1.4897× | 1.3979× | 1.3643× |
| Breast Cancer | 251 | indices | 1.9730× | 1.9481× | 1.6811× | 1.7359× |
| Breast Cancer | 251 | predict_uniform | 2.0595× | 1.9226× | 2.3728× | 2.0996× |
| Breast Cancer | 251 | proba_uniform | 1.4807× | 1.5260× | 1.6637× | 1.7194× |
| Breast Cancer | 251 | regress_uniform | 1.4994× | 1.4778× | 1.5149× | 1.6357× |
| Breast Cancer | 251 | predict_distance | 1.3830× | 1.3844× | 1.3124× | 1.2887× |
| Breast Cancer | 251 | proba_distance | 1.8873× | 1.8850× | 1.6664× | 1.8882× |
| Breast Cancer | 251 | regress_distance | 2.8033× | 2.2702× | 3.2918× | 2.9602× |
| Breast Cancer | 252 | fit_query | 1.4006× | 1.4138× | 1.4322× | 1.4755× |
| Breast Cancer | 252 | kneighbors | 1.7567× | 1.7648× | 1.5910× | 1.6801× |
| Breast Cancer | 252 | indices | 1.7210× | 1.8677× | 2.0312× | 2.0108× |
| Breast Cancer | 252 | predict_uniform | 1.4702× | 1.4892× | 1.5104× | 1.5610× |
| Breast Cancer | 252 | proba_uniform | 1.6912× | 1.5742× | 1.8859× | 1.8580× |
| Breast Cancer | 252 | regress_uniform | 1.5151× | 1.4725× | 1.3703× | 1.3810× |
| Breast Cancer | 252 | predict_distance | 1.8828× | 1.8222× | 1.4558× | 1.5773× |
| Breast Cancer | 252 | proba_distance | 1.9470× | 2.1994× | 2.2410× | 2.0666× |
| Breast Cancer | 252 | regress_distance | 1.4013× | 1.4612× | 1.4354× | 1.4678× |
| Offset +1e4, 4096×128 | 151 | kneighbors | **0.9976×** | 1.0032× | **0.9724×** | 1.0035× |
| Offset +1e4, 4096×128 | 151 | fit_query | 1.2001× | 1.0693× | 1.2008× | 1.0719× |
| Offset +1e4, 4096×128 | 251 | kneighbors | **0.9872×** | **0.9974×** | **0.9558×** | 1.0392× |
| Offset +1e4, 4096×128 | 251 | fit_query | 1.0487× | 1.0874× | 1.0822× | 1.1474× |
| Offset +1e4, 4096×128 | 252 | kneighbors | 1.0125× | **0.9759×** | 1.0157× | 1.0084× |
| Offset +1e4, 4096×128 | 252 | fit_query | 1.0053× | 1.0417× | **0.9987×** | 1.0784× |

## Accuracy and source identity

All cold, warmup, probe and retained strict audits remain in the raw evidence. Every worker passes; no accuracy failure filters a timing denominator. Each repeat fixture and oracle record exactly matches the corresponding full-control fixture. Source, runtime and harness archive bytes, measured worker identities, and current source/native/harness files have been independently verified.

| Worker | Cold passes/audits | Warmup passes/audits | Probe passes/audits | Retained passes/audits |
| --- | --- | --- | --- | --- |
| Current auto | 9/9 | 219773/219773 | 60/60 | 1860/1860 |
| Current auto A/A | 9/9 | 222025/222025 | 60/60 | 1860/1860 |
| Current forced GPU | 9/9 | 172032/172032 | 60/60 | 1860/1860 |
| Immediate prior auto | 9/9 | 182123/182123 | 60/60 | 1860/1860 |
| Historical forced GPU | 9/9 | 170638/170638 | 60/60 | 1860/1860 |

All raw paired ratios, actual routing, first/middle/last summaries, position and predecessor diagnostics remain in the independent audit. They diagnose timing sensitivity without claiming an identified clock mechanism.

## Evidence

[Every raw observation](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-0b8b3beeccd1114db2a1) — SHA-256 `19b9ea48adb3c58db720dd516a80e3fe3457cdeed144d799f644aaf384653ff1`.

[Independent audit and complete crossmatching](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-75b6542c744574b87de3).

[Full balanced control](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-7ddc36ba1227fd47229a).
