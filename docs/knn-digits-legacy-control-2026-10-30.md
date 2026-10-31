# Predeclared Digits, Breast Cancer and offset legacy-order control

All **60 operation cells**, both automatic replicas and all five workers are retained. Accuracy, runtime identity and numerical-oracle validation pass. The panel-specific crosschecks below retain both the full result and the repeat, including smaller or inconsistent slowdowns. Numerical acceptance is separate from the declared Digits performance screen.

This is the `legacy-control` portion of the [saved predeclaration](../benchmarks/results/knn-digits-final-predeclaration-2026-10-30.json) (SHA-256 `68a31e8f20c4dfe5161fd0df852ab6a208c15a73dd3099a5a5374670642b2b63`). Both nine-fixture inventories were retained from the preceding final validation and declared before candidate GPU controls and full panels. Baseline profiling or native ablations may already have occurred. The exact requested inventories are verified, with no seed or operation removed.

Each operation uses 31 serial interleaved rounds across current auto, identical current auto A/A, current forced GPU, immediate prior auto (`knn-startup-final2-2026-10-30` plus its exact archived native runtime), and historical forced GPU (`knn-small-final-2026-10-30`). This panel uses the legacy forward/reverse ordering and 1 second of continuous warmup. The full 48-fixture control uses balanced Williams ordering and 0.25 seconds of warmup. Absolute milliseconds across these runs are not directly comparable; ratios compare implementations within their own run. Neither run replaces the other, and no faster window or run is selected.

M/P means ratio of medians / median paired ratio. Ratios are comparator/current; below 1× means the automatic replica is slower and is bold. A >5% runtime slowdown means ratio <1/1.05 for BOTH statistics and BOTH independent automatic replicas. This recurrence screen is descriptive, not a confidence interval or significance test.

## Full-control exceptions and repeat crosscheck

| Comparator | Flags in full 222 cells | Covered here | Recurring | Not recurring | New flags |
| --- | --- | --- | --- | --- | --- |
| Immediate prior auto | 18 | 9 | 3 | 6 | 8 |
| Current forced GPU | 1 | 0 | 0 | 0 | 0 |
| Historical forced GPU | 0 | 0 | 0 | 0 | 0 |

| Comparator | Classification | Family | Seed | Operation | Full current M/P | Full A/A M/P | Repeat current M/P | Repeat A/A M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Immediate prior auto | recurring | Breast Cancer | 151 | predict_distance | **0.9311×** / **0.9058×** | **0.9312×** / **0.9228×** | **0.8510×** / **0.8695×** | **0.8546×** / **0.8472×** |
| Immediate prior auto | recurring | Breast Cancer | 251 | regress_uniform | **0.9012×** / **0.9259×** | **0.9349×** / **0.9381×** | **0.8261×** / **0.8028×** | **0.8369×** / **0.8003×** |
| Immediate prior auto | recurring | Breast Cancer | 252 | proba_uniform | **0.9450×** / **0.9079×** | **0.9342×** / **0.9315×** | **0.8027×** / **0.8159×** | **0.7914×** / **0.8315×** |
| Immediate prior auto | not recurring | Breast Cancer | 151 | regress_uniform | **0.9307×** / **0.9127×** | **0.9381×** / **0.9410×** | **0.8675×** / **0.9842×** | **0.8466×** / **0.9472×** |
| Immediate prior auto | not recurring | Breast Cancer | 251 | proba_uniform | **0.8560×** / **0.8407×** | **0.9487×** / **0.9336×** | **0.9547×** / **0.9500×** | **0.9796×** / **0.9921×** |
| Immediate prior auto | not recurring | Breast Cancer | 252 | regress_distance | **0.8962×** / **0.8646×** | **0.8701×** / **0.8904×** | **0.9516×** / **0.8820×** | **0.9921×** / **0.9739×** |
| Immediate prior auto | not recurring | Digits | 151 | predict_uniform | **0.8914×** / **0.8972×** | **0.9027×** / **0.9381×** | 1.0998× / 1.0191× | 1.0786× / 1.0325× |
| Immediate prior auto | not recurring | Digits | 151 | regress_distance | **0.8672×** / **0.8759×** | **0.8860×** / **0.8587×** | **0.9833×** / 1.0235× | **0.9910×** / **0.9733×** |
| Immediate prior auto | not recurring | Digits | 252 | predict_distance | **0.8300×** / **0.8524×** | **0.8923×** / **0.8939×** | 1.0829× / 1.0759× | 1.1056× / 1.1376× |
| Immediate prior auto | new in repeat | Breast Cancer | 151 | kneighbors | **0.9783×** / **0.9682×** | **0.9639×** / **0.9967×** | **0.7882×** / **0.8093×** | **0.7764×** / **0.8512×** |
| Immediate prior auto | new in repeat | Breast Cancer | 251 | fit_query | **0.9349×** / **0.9554×** | 1.0663× / 1.0524× | **0.8872×** / **0.8862×** | **0.8636×** / **0.8667×** |
| Immediate prior auto | new in repeat | Breast Cancer | 252 | predict_uniform | **0.8732×** / **0.8855×** | **0.9976×** / **0.9692×** | **0.8598×** / **0.9106×** | **0.9004×** / **0.9505×** |
| Immediate prior auto | new in repeat | Digits | 151 | proba_uniform | 1.0346× / 1.0603× | 1.0566× / 1.0259× | **0.8846×** / **0.9006×** | **0.8683×** / **0.9094×** |
| Immediate prior auto | new in repeat | Digits | 251 | predict_uniform | 1.0927× / 1.0783× | 1.1363× / 1.1071× | **0.8581×** / **0.8603×** | **0.8604×** / **0.8433×** |
| Immediate prior auto | new in repeat | Digits | 251 | regress_distance | **0.9514×** / **0.9234×** | 1.0073× / **0.9896×** | **0.9260×** / **0.9100×** | **0.9035×** / **0.8330×** |
| Immediate prior auto | new in repeat | Digits | 252 | indices | 1.0400× / 1.0012× | 1.0792× / 1.0780× | **0.9156×** / **0.8945×** | **0.9228×** / **0.9326×** |
| Immediate prior auto | new in repeat | Digits | 252 | proba_distance | **0.9903×** / 1.0099× | **0.9656×** / **0.9960×** | **0.8931×** / **0.8984×** | **0.8769×** / **0.9202×** |

## Every slower observation in either statistic

| Comparator | Replica | Family | Seed | Operation | M/P |
| --- | --- | --- | --- | --- | --- |
| Immediate prior auto | Current auto | Digits | 151 | proba_uniform | **0.8846×** / **0.9006×** |
| Immediate prior auto | Current auto A/A | Digits | 151 | proba_uniform | **0.8683×** / **0.9094×** |
| Immediate prior auto | Current auto | Digits | 151 | regress_distance | **0.9833×** / 1.0235× |
| Immediate prior auto | Current auto A/A | Digits | 151 | regress_distance | **0.9910×** / **0.9733×** |
| Immediate prior auto | Current auto | Digits | 251 | predict_uniform | **0.8581×** / **0.8603×** |
| Immediate prior auto | Current auto A/A | Digits | 251 | predict_uniform | **0.8604×** / **0.8433×** |
| Immediate prior auto | Current auto | Digits | 251 | regress_distance | **0.9260×** / **0.9100×** |
| Immediate prior auto | Current auto A/A | Digits | 251 | regress_distance | **0.9035×** / **0.8330×** |
| Immediate prior auto | Current auto | Digits | 252 | indices | **0.9156×** / **0.8945×** |
| Immediate prior auto | Current auto A/A | Digits | 252 | indices | **0.9228×** / **0.9326×** |
| Immediate prior auto | Current auto | Digits | 252 | proba_distance | **0.8931×** / **0.8984×** |
| Immediate prior auto | Current auto A/A | Digits | 252 | proba_distance | **0.8769×** / **0.9202×** |
| Immediate prior auto | Current auto | Breast Cancer | 151 | fit_query | **0.9344×** / 1.0378× |
| Immediate prior auto | Current auto A/A | Breast Cancer | 151 | fit_query | **0.9833×** / **0.9834×** |
| Immediate prior auto | Current auto | Breast Cancer | 151 | kneighbors | **0.7882×** / **0.8093×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 151 | kneighbors | **0.7764×** / **0.8512×** |
| Immediate prior auto | Current auto | Breast Cancer | 151 | indices | **0.9127×** / **0.9375×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 151 | indices | **0.9895×** / **0.9918×** |
| Immediate prior auto | Current auto | Breast Cancer | 151 | regress_uniform | **0.8675×** / **0.9842×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 151 | regress_uniform | **0.8466×** / **0.9472×** |
| Immediate prior auto | Current auto | Breast Cancer | 151 | predict_distance | **0.8510×** / **0.8695×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 151 | predict_distance | **0.8546×** / **0.8472×** |
| Immediate prior auto | Current auto | Breast Cancer | 151 | proba_distance | **0.9929×** / **0.9723×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 151 | proba_distance | **0.9866×** / **0.9629×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 151 | regress_distance | **0.9320×** / **0.9795×** |
| Immediate prior auto | Current auto | Breast Cancer | 251 | fit_query | **0.8872×** / **0.8862×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 251 | fit_query | **0.8636×** / **0.8667×** |
| Immediate prior auto | Current auto | Breast Cancer | 251 | predict_uniform | **0.9601×** / **1.0000×** |
| Immediate prior auto | Current auto | Breast Cancer | 251 | proba_uniform | **0.9547×** / **0.9500×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 251 | proba_uniform | **0.9796×** / **0.9921×** |
| Immediate prior auto | Current auto | Breast Cancer | 251 | regress_uniform | **0.8261×** / **0.8028×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 251 | regress_uniform | **0.8369×** / **0.8003×** |
| Immediate prior auto | Current auto | Breast Cancer | 251 | predict_distance | **0.8307×** / **0.9090×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 251 | predict_distance | **0.9457×** / **0.9866×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 251 | proba_distance | **0.9691×** / **0.9926×** |
| Immediate prior auto | Current auto | Breast Cancer | 251 | regress_distance | **0.9613×** / **0.9634×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 251 | regress_distance | 1.0379× / **0.9927×** |
| Immediate prior auto | Current auto | Breast Cancer | 252 | fit_query | **0.9770×** / **0.9981×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 252 | fit_query | **0.9624×** / **0.9795×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 252 | kneighbors | **0.9966×** / **0.9536×** |
| Immediate prior auto | Current auto | Breast Cancer | 252 | predict_uniform | **0.8598×** / **0.9106×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 252 | predict_uniform | **0.9004×** / **0.9505×** |
| Immediate prior auto | Current auto | Breast Cancer | 252 | proba_uniform | **0.8027×** / **0.8159×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 252 | proba_uniform | **0.7914×** / **0.8315×** |
| Immediate prior auto | Current auto | Breast Cancer | 252 | regress_uniform | **0.9417×** / **0.9543×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 252 | regress_uniform | **0.8716×** / 1.0014× |
| Immediate prior auto | Current auto A/A | Breast Cancer | 252 | predict_distance | 1.0116× / **0.9967×** |
| Immediate prior auto | Current auto | Breast Cancer | 252 | proba_distance | **0.9603×** / **0.9635×** |
| Immediate prior auto | Current auto | Breast Cancer | 252 | regress_distance | **0.9516×** / **0.8820×** |
| Immediate prior auto | Current auto A/A | Breast Cancer | 252 | regress_distance | **0.9921×** / **0.9739×** |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 151 | kneighbors | 1.0097× / **0.9941×** |
| Historical forced GPU | Current auto | Offset +1e4, 4096×128 | 151 | kneighbors | 1.0187× / **0.9986×** |
| Immediate prior auto | Current auto | Offset +1e4, 4096×128 | 151 | fit_query | **0.9811×** / 1.0014× |
| Historical forced GPU | Current auto | Offset +1e4, 4096×128 | 151 | fit_query | **0.9869×** / 1.0626× |
| Immediate prior auto | Current auto | Offset +1e4, 4096×128 | 251 | kneighbors | 1.0320× / **0.9966×** |
| Historical forced GPU | Current auto | Offset +1e4, 4096×128 | 251 | kneighbors | 1.0074× / **0.9950×** |
| Historical forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 251 | kneighbors | **0.9999×** / 1.0092× |
| Immediate prior auto | Current auto | Offset +1e4, 4096×128 | 251 | fit_query | **0.9820×** / **0.9547×** |
| Immediate prior auto | Current auto A/A | Offset +1e4, 4096×128 | 251 | fit_query | **0.9541×** / **0.9526×** |
| Current forced GPU | Current auto | Offset +1e4, 4096×128 | 251 | fit_query | **0.9840×** / **0.9764×** |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 251 | fit_query | **0.9561×** / **0.9532×** |
| Current forced GPU | Current auto | Offset +1e4, 4096×128 | 252 | kneighbors | **0.9679×** / **0.9504×** |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 252 | kneighbors | **0.9710×** / **0.9596×** |
| Historical forced GPU | Current auto | Offset +1e4, 4096×128 | 252 | kneighbors | **0.9908×** / 1.0001× |
| Historical forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 252 | kneighbors | **0.9940×** / 1.0296× |
| Immediate prior auto | Current auto | Offset +1e4, 4096×128 | 252 | fit_query | 1.0419× / **0.9593×** |
| Immediate prior auto | Current auto A/A | Offset +1e4, 4096×128 | 252 | fit_query | 1.0713× / **0.9844×** |
| Current forced GPU | Current auto | Offset +1e4, 4096×128 | 252 | fit_query | 1.0137× / **0.9831×** |
| Current forced GPU | Current auto A/A | Offset +1e4, 4096×128 | 252 | fit_query | 1.0422× / **0.9819×** |

## Every repeat timing cell

All times are median milliseconds from this repeat only.

| Family | Seed | Operation | Current auto ms | Current auto A/A ms | Current forced GPU ms | Immediate prior auto ms | Historical forced GPU ms | A/A over auto M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 0.502166 | 0.517625 | 0.997167 | 0.565584 | 0.639333 | 1.0308× / **0.9934×** |
| Digits | 151 | kneighbors | 0.285625 | 0.328750 | 0.979625 | 0.380458 | 0.900750 | 1.1510× / 1.1589× |
| Digits | 151 | indices | 0.356458 | 0.273916 | 0.883291 | 0.403584 | 0.911917 | **0.7684×** / **0.8086×** |
| Digits | 151 | predict_uniform | 0.394958 | 0.402708 | 0.502250 | 0.434375 | 0.511000 | 1.0196× / **0.9915×** |
| Digits | 151 | proba_uniform | 0.366917 | 0.373833 | 0.939334 | 0.324583 | 0.905167 | 1.0188× / **0.9772×** |
| Digits | 151 | regress_uniform | 0.320625 | 0.348250 | 0.508708 | 0.358083 | 0.410958 | 1.0862× / 1.0643× |
| Digits | 151 | predict_distance | 0.418166 | 0.448250 | 0.571833 | 0.491292 | 0.809250 | 1.0719× / 1.0063× |
| Digits | 151 | proba_distance | 0.453916 | 0.449708 | 1.052875 | 0.503583 | 0.919333 | **0.9907×** / **0.9846×** |
| Digits | 151 | regress_distance | 0.354250 | 0.351500 | 0.439833 | 0.348333 | 0.417958 | **0.9922×** / **0.9836×** |
| Digits | 251 | fit_query | 0.481458 | 0.494916 | 1.047875 | 0.552416 | 1.026250 | 1.0280× / 1.0197× |
| Digits | 251 | kneighbors | 0.371542 | 0.306416 | 0.476416 | 0.402667 | 0.832250 | **0.8247×** / **0.8515×** |
| Digits | 251 | indices | 0.287583 | 0.299958 | 0.422791 | 0.352583 | 0.430166 | 1.0430× / **0.9993×** |
| Digits | 251 | predict_uniform | 0.370958 | 0.370000 | 0.508542 | 0.318333 | 0.507291 | **0.9974×** / 1.0413× |
| Digits | 251 | proba_uniform | 0.409208 | 0.392458 | 0.504583 | 0.414750 | 0.470750 | **0.9591×** / **0.9945×** |
| Digits | 251 | regress_uniform | 0.283458 | 0.337708 | 0.988458 | 0.391375 | 0.906375 | 1.1914× / 1.2443× |
| Digits | 251 | predict_distance | 0.436458 | 0.394833 | 0.553917 | 0.493083 | 0.880250 | **0.9046×** / **0.8595×** |
| Digits | 251 | proba_distance | 0.364709 | 0.352417 | 0.449083 | 0.379000 | 0.459125 | **0.9663×** / **0.9371×** |
| Digits | 251 | regress_distance | 0.395917 | 0.405792 | 0.702916 | 0.366625 | 0.494250 | 1.0249× / 1.0187× |
| Digits | 252 | fit_query | 0.461250 | 0.396292 | 1.039459 | 0.506000 | 1.115000 | **0.8592×** / **0.8671×** |
| Digits | 252 | kneighbors | 0.324375 | 0.324333 | 0.409291 | 0.378750 | 0.914292 | **0.9999×** / **0.9628×** |
| Digits | 252 | indices | 0.347709 | 0.345000 | 1.021458 | 0.318375 | 1.004625 | **0.9922×** / 1.0021× |
| Digits | 252 | predict_uniform | 0.311458 | 0.322667 | 0.435292 | 0.355667 | 0.407416 | 1.0360× / 1.0833× |
| Digits | 252 | proba_uniform | 0.295708 | 0.362375 | 0.988541 | 0.419250 | 0.959750 | 1.2254× / 1.2496× |
| Digits | 252 | regress_uniform | 0.400167 | 0.341708 | 0.800917 | 0.445500 | 0.824416 | **0.8539×** / **0.8769×** |
| Digits | 252 | predict_distance | 0.418416 | 0.409792 | 0.448667 | 0.453083 | 0.455917 | **0.9794×** / 1.0027× |
| Digits | 252 | proba_distance | 0.391958 | 0.399167 | 0.417916 | 0.350041 | 0.441417 | 1.0184× / 1.0186× |
| Digits | 252 | regress_distance | 0.444750 | 0.429250 | 0.607833 | 0.455625 | 0.480041 | **0.9651×** / **0.9681×** |
| Breast Cancer | 151 | fit_query | 0.277625 | 0.263833 | 0.329416 | 0.259417 | 0.383834 | **0.9503×** / **0.9982×** |
| Breast Cancer | 151 | kneighbors | 0.175875 | 0.178542 | 0.277209 | 0.138625 | 0.481708 | 1.0152× / **0.9885×** |
| Breast Cancer | 151 | indices | 0.154583 | 0.142583 | 0.235417 | 0.141083 | 0.204084 | **0.9224×** / **0.9980×** |
| Breast Cancer | 151 | predict_uniform | 0.149708 | 0.157667 | 0.303000 | 0.171916 | 0.437333 | 1.0532× / 1.0797× |
| Breast Cancer | 151 | proba_uniform | 0.199250 | 0.174708 | 0.309083 | 0.202750 | 0.342167 | **0.8768×** / **0.9074×** |
| Breast Cancer | 151 | regress_uniform | 0.199083 | 0.204000 | 0.244375 | 0.172708 | 0.263792 | 1.0247× / **0.9706×** |
| Breast Cancer | 151 | predict_distance | 0.206625 | 0.205750 | 0.448750 | 0.175834 | 0.293875 | **0.9958×** / 1.0062× |
| Breast Cancer | 151 | proba_distance | 0.194083 | 0.195333 | 0.274584 | 0.192708 | 0.254959 | 1.0064× / 1.0185× |
| Breast Cancer | 151 | regress_distance | 0.161833 | 0.185583 | 0.479833 | 0.172958 | 0.487459 | 1.1468× / 1.1398× |
| Breast Cancer | 251 | fit_query | 0.210583 | 0.216333 | 0.325875 | 0.186834 | 0.576208 | 1.0273× / 1.0015× |
| Breast Cancer | 251 | kneighbors | 0.176167 | 0.170375 | 0.258916 | 0.180667 | 0.225667 | **0.9671×** / 1.0067× |
| Breast Cancer | 251 | indices | 0.133125 | 0.151167 | 0.273250 | 0.164167 | 0.256125 | 1.1355× / 1.1145× |
| Breast Cancer | 251 | predict_uniform | 0.181500 | 0.164458 | 0.292792 | 0.174250 | 0.282041 | **0.9061×** / **0.8848×** |
| Breast Cancer | 251 | proba_uniform | 0.173750 | 0.169333 | 0.226458 | 0.165875 | 0.245792 | **0.9746×** / **0.9565×** |
| Breast Cancer | 251 | regress_uniform | 0.170583 | 0.168375 | 0.511125 | 0.140917 | 0.401875 | **0.9871×** / **0.9837×** |
| Breast Cancer | 251 | predict_distance | 0.201084 | 0.176625 | 0.276042 | 0.167042 | 0.259583 | **0.8784×** / **0.9408×** |
| Breast Cancer | 251 | proba_distance | 0.164292 | 0.190125 | 0.498333 | 0.184250 | 0.528833 | 1.1572× / 1.1094× |
| Breast Cancer | 251 | regress_distance | 0.193584 | 0.179292 | 0.533417 | 0.186083 | 0.529833 | **0.9262×** / **0.9596×** |
| Breast Cancer | 252 | fit_query | 0.233833 | 0.237375 | 0.356541 | 0.228458 | 0.327417 | 1.0151× / **0.9841×** |
| Breast Cancer | 252 | kneighbors | 0.150542 | 0.172833 | 0.262292 | 0.172250 | 0.253208 | 1.1481× / 1.1668× |
| Breast Cancer | 252 | indices | 0.145833 | 0.138459 | 0.469917 | 0.153209 | 0.496500 | **0.9494×** / **0.9313×** |
| Breast Cancer | 252 | predict_uniform | 0.171792 | 0.164041 | 0.248458 | 0.147708 | 0.258250 | **0.9549×** / **0.9416×** |
| Breast Cancer | 252 | proba_uniform | 0.176291 | 0.178792 | 0.547750 | 0.141500 | 0.502167 | 1.0142× / 1.0410× |
| Breast Cancer | 252 | regress_uniform | 0.167250 | 0.180708 | 0.254292 | 0.157500 | 0.234584 | 1.0805× / **0.9635×** |
| Breast Cancer | 252 | predict_distance | 0.170209 | 0.175541 | 0.559000 | 0.177584 | 0.512666 | 1.0313× / 1.0931× |
| Breast Cancer | 252 | proba_distance | 0.178541 | 0.165333 | 0.511000 | 0.171458 | 0.534333 | **0.9260×** / **0.9183×** |
| Breast Cancer | 252 | regress_distance | 0.192750 | 0.184875 | 0.264042 | 0.183417 | 0.263209 | **0.9591×** / **0.9567×** |
| Offset +1e4, 4096×128 | 151 | kneighbors | 0.375292 | 0.378875 | 0.382542 | 0.386167 | 0.382292 | 1.0095× / 1.0160× |
| Offset +1e4, 4096×128 | 151 | fit_query | 0.670792 | 0.640500 | 0.672250 | 0.658125 | 0.662000 | **0.9548×** / **0.9582×** |
| Offset +1e4, 4096×128 | 251 | kneighbors | 0.377625 | 0.380459 | 0.396750 | 0.389709 | 0.380417 | 1.0075× / **0.9691×** |
| Offset +1e4, 4096×128 | 251 | fit_query | 0.642625 | 0.661375 | 0.632333 | 0.631042 | 0.686583 | 1.0292× / 1.0272× |
| Offset +1e4, 4096×128 | 252 | kneighbors | 0.391042 | 0.389791 | 0.378500 | 0.405250 | 0.387458 | **0.9968×** / **0.9727×** |
| Offset +1e4, 4096×128 | 252 | fit_query | 0.635917 | 0.618500 | 0.644625 | 0.662583 | 0.694084 | **0.9726×** / **0.9936×** |

## Every cell against Immediate prior auto

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.1263× | 1.0999× | 1.0927× | 1.0868× |
| Digits | 151 | kneighbors | 1.3320× | 1.2842× | 1.1573× | 1.1340× |
| Digits | 151 | indices | 1.1322× | 1.1641× | 1.4734× | 1.4700× |
| Digits | 151 | predict_uniform | 1.0998× | 1.0191× | 1.0786× | 1.0325× |
| Digits | 151 | proba_uniform | **0.8846×** | **0.9006×** | **0.8683×** | **0.9094×** |
| Digits | 151 | regress_uniform | 1.1168× | 1.1226× | 1.0282× | 1.0364× |
| Digits | 151 | predict_distance | 1.1749× | 1.2243× | 1.0960× | 1.1131× |
| Digits | 151 | proba_distance | 1.1094× | 1.0893× | 1.1198× | 1.1063× |
| Digits | 151 | regress_distance | **0.9833×** | 1.0235× | **0.9910×** | **0.9733×** |
| Digits | 251 | fit_query | 1.1474× | 1.1454× | 1.1162× | 1.1449× |
| Digits | 251 | kneighbors | 1.0838× | 1.1100× | 1.3141× | 1.3141× |
| Digits | 251 | indices | 1.2260× | 1.1832× | 1.1754× | 1.1171× |
| Digits | 251 | predict_uniform | **0.8581×** | **0.8603×** | **0.8604×** | **0.8433×** |
| Digits | 251 | proba_uniform | 1.0135× | 1.0369× | 1.0568× | 1.0555× |
| Digits | 251 | regress_uniform | 1.3807× | 1.3356× | 1.1589× | 1.1459× |
| Digits | 251 | predict_distance | 1.1297× | 1.0797× | 1.2488× | 1.2620× |
| Digits | 251 | proba_distance | 1.0392× | 1.0618× | 1.0754× | 1.1454× |
| Digits | 251 | regress_distance | **0.9260×** | **0.9100×** | **0.9035×** | **0.8330×** |
| Digits | 252 | fit_query | 1.0970× | 1.1073× | 1.2768× | 1.2598× |
| Digits | 252 | kneighbors | 1.1676× | 1.1445× | 1.1678× | 1.2026× |
| Digits | 252 | indices | **0.9156×** | **0.8945×** | **0.9228×** | **0.9326×** |
| Digits | 252 | predict_uniform | 1.1419× | 1.1363× | 1.1023× | 1.0735× |
| Digits | 252 | proba_uniform | 1.4178× | 1.3821× | 1.1570× | 1.1449× |
| Digits | 252 | regress_uniform | 1.1133× | 1.0873× | 1.3037× | 1.2881× |
| Digits | 252 | predict_distance | 1.0829× | 1.0759× | 1.1056× | 1.1376× |
| Digits | 252 | proba_distance | **0.8931×** | **0.8984×** | **0.8769×** | **0.9202×** |
| Digits | 252 | regress_distance | 1.0245× | 1.0400× | 1.0614× | 1.0657× |
| Breast Cancer | 151 | fit_query | **0.9344×** | 1.0378× | **0.9833×** | **0.9834×** |
| Breast Cancer | 151 | kneighbors | **0.7882×** | **0.8093×** | **0.7764×** | **0.8512×** |
| Breast Cancer | 151 | indices | **0.9127×** | **0.9375×** | **0.9895×** | **0.9918×** |
| Breast Cancer | 151 | predict_uniform | 1.1483× | 1.1383× | 1.0904× | 1.0150× |
| Breast Cancer | 151 | proba_uniform | 1.0176× | 1.0336× | 1.1605× | 1.1311× |
| Breast Cancer | 151 | regress_uniform | **0.8675×** | **0.9842×** | **0.8466×** | **0.9472×** |
| Breast Cancer | 151 | predict_distance | **0.8510×** | **0.8695×** | **0.8546×** | **0.8472×** |
| Breast Cancer | 151 | proba_distance | **0.9929×** | **0.9723×** | **0.9866×** | **0.9629×** |
| Breast Cancer | 151 | regress_distance | 1.0687× | 1.0782× | **0.9320×** | **0.9795×** |
| Breast Cancer | 251 | fit_query | **0.8872×** | **0.8862×** | **0.8636×** | **0.8667×** |
| Breast Cancer | 251 | kneighbors | 1.0255× | 1.0039× | 1.0604× | 1.0257× |
| Breast Cancer | 251 | indices | 1.2332× | 1.2303× | 1.0860× | 1.0394× |
| Breast Cancer | 251 | predict_uniform | **0.9601×** | **1.0000×** | 1.0595× | 1.0862× |
| Breast Cancer | 251 | proba_uniform | **0.9547×** | **0.9500×** | **0.9796×** | **0.9921×** |
| Breast Cancer | 251 | regress_uniform | **0.8261×** | **0.8028×** | **0.8369×** | **0.8003×** |
| Breast Cancer | 251 | predict_distance | **0.8307×** | **0.9090×** | **0.9457×** | **0.9866×** |
| Breast Cancer | 251 | proba_distance | 1.1215× | 1.1582× | **0.9691×** | **0.9926×** |
| Breast Cancer | 251 | regress_distance | **0.9613×** | **0.9634×** | 1.0379× | **0.9927×** |
| Breast Cancer | 252 | fit_query | **0.9770×** | **0.9981×** | **0.9624×** | **0.9795×** |
| Breast Cancer | 252 | kneighbors | 1.1442× | 1.1153× | **0.9966×** | **0.9536×** |
| Breast Cancer | 252 | indices | 1.0506× | 1.0764× | 1.1065× | 1.1063× |
| Breast Cancer | 252 | predict_uniform | **0.8598×** | **0.9106×** | **0.9004×** | **0.9505×** |
| Breast Cancer | 252 | proba_uniform | **0.8027×** | **0.8159×** | **0.7914×** | **0.8315×** |
| Breast Cancer | 252 | regress_uniform | **0.9417×** | **0.9543×** | **0.8716×** | 1.0014× |
| Breast Cancer | 252 | predict_distance | 1.0433× | 1.0738× | 1.0116× | **0.9967×** |
| Breast Cancer | 252 | proba_distance | **0.9603×** | **0.9635×** | 1.0370× | 1.0830× |
| Breast Cancer | 252 | regress_distance | **0.9516×** | **0.8820×** | **0.9921×** | **0.9739×** |
| Offset +1e4, 4096×128 | 151 | kneighbors | 1.0290× | 1.0387× | 1.0192× | 1.0177× |
| Offset +1e4, 4096×128 | 151 | fit_query | **0.9811×** | 1.0014× | 1.0275× | 1.0646× |
| Offset +1e4, 4096×128 | 251 | kneighbors | 1.0320× | **0.9966×** | 1.0243× | 1.0259× |
| Offset +1e4, 4096×128 | 251 | fit_query | **0.9820×** | **0.9547×** | **0.9541×** | **0.9526×** |
| Offset +1e4, 4096×128 | 252 | kneighbors | 1.0363× | 1.0056× | 1.0397× | 1.0281× |
| Offset +1e4, 4096×128 | 252 | fit_query | 1.0419× | **0.9593×** | 1.0713× | **0.9844×** |

## Every cell against Current forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.9857× | 1.9857× | 1.9264× | 1.9326× |
| Digits | 151 | kneighbors | 3.4298× | 3.5610× | 2.9798× | 2.9628× |
| Digits | 151 | indices | 2.4780× | 2.3183× | 3.2247× | 2.6942× |
| Digits | 151 | predict_uniform | 1.2717× | 1.3602× | 1.2472× | 1.3193× |
| Digits | 151 | proba_uniform | 2.5601× | 2.6495× | 2.5127× | 2.6825× |
| Digits | 151 | regress_uniform | 1.5866× | 1.8748× | 1.4608× | 1.7318× |
| Digits | 151 | predict_distance | 1.3675× | 1.5148× | 1.2757× | 1.2890× |
| Digits | 151 | proba_distance | 2.3195× | 2.0378× | 2.3412× | 1.9877× |
| Digits | 151 | regress_distance | 1.2416× | 1.3016× | 1.2513× | 1.2450× |
| Digits | 251 | fit_query | 2.1765× | 2.1442× | 2.1173× | 2.2698× |
| Digits | 251 | kneighbors | 1.2823× | 1.5017× | 1.5548× | 1.7849× |
| Digits | 251 | indices | 1.4702× | 1.5250× | 1.4095× | 1.4676× |
| Digits | 251 | predict_uniform | 1.3709× | 1.5756× | 1.3744× | 1.5327× |
| Digits | 251 | proba_uniform | 1.2331× | 1.2916× | 1.2857× | 1.3583× |
| Digits | 251 | regress_uniform | 3.4871× | 3.1388× | 2.9270× | 2.5748× |
| Digits | 251 | predict_distance | 1.2691× | 1.3578× | 1.4029× | 1.4102× |
| Digits | 251 | proba_distance | 1.2313× | 1.2422× | 1.2743× | 1.4040× |
| Digits | 251 | regress_distance | 1.7754× | 1.9814× | 1.7322× | 1.5600× |
| Digits | 252 | fit_query | 2.2536× | 2.3216× | 2.6230× | 2.4708× |
| Digits | 252 | kneighbors | 1.2618× | 1.4086× | 1.2619× | 1.3301× |
| Digits | 252 | indices | 2.9377× | 3.1948× | 2.9607× | 3.0881× |
| Digits | 252 | predict_uniform | 1.3976× | 1.6137× | 1.3490× | 1.4917× |
| Digits | 252 | proba_uniform | 3.3430× | 3.1287× | 2.7280× | 2.7918× |
| Digits | 252 | regress_uniform | 2.0015× | 2.0898× | 2.3439× | 2.2820× |
| Digits | 252 | predict_distance | 1.0723× | 1.1239× | 1.0949× | 1.0874× |
| Digits | 252 | proba_distance | 1.0662× | 1.3113× | 1.0470× | 1.1780× |
| Digits | 252 | regress_distance | 1.3667× | 1.4058× | 1.4160× | 1.4940× |
| Breast Cancer | 151 | fit_query | 1.1866× | 1.2575× | 1.2486× | 1.2510× |
| Breast Cancer | 151 | kneighbors | 1.5762× | 1.6587× | 1.5526× | 1.7652× |
| Breast Cancer | 151 | indices | 1.5229× | 1.5674× | 1.6511× | 1.5639× |
| Breast Cancer | 151 | predict_uniform | 2.0239× | 2.1184× | 1.9218× | 1.9478× |
| Breast Cancer | 151 | proba_uniform | 1.5512× | 1.6829× | 1.7691× | 1.8570× |
| Breast Cancer | 151 | regress_uniform | 1.2275× | 1.2247× | 1.1979× | 1.2708× |
| Breast Cancer | 151 | predict_distance | 2.1718× | 1.9719× | 2.1810× | 2.1118× |
| Breast Cancer | 151 | proba_distance | 1.4148× | 1.5313× | 1.4057× | 1.4843× |
| Breast Cancer | 151 | regress_distance | 2.9650× | 2.5821× | 2.5855× | 2.1304× |
| Breast Cancer | 251 | fit_query | 1.5475× | 1.6151× | 1.5064× | 1.6808× |
| Breast Cancer | 251 | kneighbors | 1.4697× | 1.5781× | 1.5197× | 1.5327× |
| Breast Cancer | 251 | indices | 2.0526× | 2.0492× | 1.8076× | 1.7965× |
| Breast Cancer | 251 | predict_uniform | 1.6132× | 1.6029× | 1.7803× | 1.7920× |
| Breast Cancer | 251 | proba_uniform | 1.3034× | 1.2522× | 1.3374× | 1.3291× |
| Breast Cancer | 251 | regress_uniform | 2.9963× | 2.5415× | 3.0356× | 2.5970× |
| Breast Cancer | 251 | predict_distance | 1.3728× | 1.4075× | 1.5629× | 1.5261× |
| Breast Cancer | 251 | proba_distance | 3.0332× | 3.1386× | 2.6211× | 2.3716× |
| Breast Cancer | 251 | regress_distance | 2.7555× | 2.2976× | 2.9751× | 2.3741× |
| Breast Cancer | 252 | fit_query | 1.5248× | 1.5126× | 1.5020× | 1.6499× |
| Breast Cancer | 252 | kneighbors | 1.7423× | 1.7342× | 1.5176× | 1.6295× |
| Breast Cancer | 252 | indices | 3.2223× | 2.3195× | 3.3939× | 2.7187× |
| Breast Cancer | 252 | predict_uniform | 1.4463× | 1.4001× | 1.5146× | 1.5739× |
| Breast Cancer | 252 | proba_uniform | 3.1071× | 2.7349× | 3.0636× | 2.6271× |
| Breast Cancer | 252 | regress_uniform | 1.5204× | 1.5217× | 1.4072× | 1.4769× |
| Breast Cancer | 252 | predict_distance | 3.2842× | 3.2437× | 3.1844× | 2.5391× |
| Breast Cancer | 252 | proba_distance | 2.8621× | 2.3035× | 3.0907× | 2.7126× |
| Breast Cancer | 252 | regress_distance | 1.3699× | 1.3867× | 1.4282× | 1.4358× |
| Offset +1e4, 4096×128 | 151 | kneighbors | 1.0193× | 1.0084× | 1.0097× | **0.9941×** |
| Offset +1e4, 4096×128 | 151 | fit_query | 1.0022× | 1.0016× | 1.0496× | 1.1473× |
| Offset +1e4, 4096×128 | 251 | kneighbors | 1.0506× | 1.0116× | 1.0428× | 1.0459× |
| Offset +1e4, 4096×128 | 251 | fit_query | **0.9840×** | **0.9764×** | **0.9561×** | **0.9532×** |
| Offset +1e4, 4096×128 | 252 | kneighbors | **0.9679×** | **0.9504×** | **0.9710×** | **0.9596×** |
| Offset +1e4, 4096×128 | 252 | fit_query | 1.0137× | **0.9831×** | 1.0422× | **0.9819×** |

## Every cell against Historical forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.2732× | 1.2831× | 1.2351× | 1.2587× |
| Digits | 151 | kneighbors | 3.1536× | 3.0092× | 2.7399× | 2.7159× |
| Digits | 151 | indices | 2.5583× | 2.6654× | 3.3292× | 3.1694× |
| Digits | 151 | predict_uniform | 1.2938× | 1.3410× | 1.2689× | 1.3977× |
| Digits | 151 | proba_uniform | 2.4670× | 2.4329× | 2.4213× | 2.3868× |
| Digits | 151 | regress_uniform | 1.2817× | 1.2286× | 1.1801× | 1.2293× |
| Digits | 151 | predict_distance | 1.9352× | 1.6755× | 1.8054× | 1.6679× |
| Digits | 151 | proba_distance | 2.0253× | 2.1124× | 2.0443× | 2.1448× |
| Digits | 151 | regress_distance | 1.1798× | 1.2359× | 1.1891× | 1.2175× |
| Digits | 251 | fit_query | 2.1315× | 2.1988× | 2.0736× | 2.3582× |
| Digits | 251 | kneighbors | 2.2400× | 2.3294× | 2.7161× | 2.8155× |
| Digits | 251 | indices | 1.4958× | 1.5127× | 1.4341× | 1.4857× |
| Digits | 251 | predict_uniform | 1.3675× | 1.4242× | 1.3711× | 1.5765× |
| Digits | 251 | proba_uniform | 1.1504× | 1.1899× | 1.1995× | 1.1786× |
| Digits | 251 | regress_uniform | 3.1976× | 2.7651× | 2.6839× | 2.5795× |
| Digits | 251 | predict_distance | 2.0168× | 2.0956× | 2.2294× | 2.0388× |
| Digits | 251 | proba_distance | 1.2589× | 1.2463× | 1.3028× | 1.4454× |
| Digits | 251 | regress_distance | 1.2484× | 1.3408× | 1.2180× | 1.3328× |
| Digits | 252 | fit_query | 2.4173× | 2.4797× | 2.8136× | 2.7955× |
| Digits | 252 | kneighbors | 2.8186× | 2.4358× | 2.8190× | 2.7324× |
| Digits | 252 | indices | 2.8893× | 2.8650× | 2.9120× | 2.7620× |
| Digits | 252 | predict_uniform | 1.3081× | 1.3069× | 1.2627× | 1.2774× |
| Digits | 252 | proba_uniform | 3.2456× | 3.0521× | 2.6485× | 2.5278× |
| Digits | 252 | regress_uniform | 2.0602× | 2.0859× | 2.4126× | 2.4058× |
| Digits | 252 | predict_distance | 1.0896× | 1.1365× | 1.1126× | 1.1611× |
| Digits | 252 | proba_distance | 1.1262× | 1.3647× | 1.1058× | 1.3689× |
| Digits | 252 | regress_distance | 1.0794× | 1.0514× | 1.1183× | 1.1446× |
| Breast Cancer | 151 | fit_query | 1.3826× | 1.5054× | 1.4548× | 1.4535× |
| Breast Cancer | 151 | kneighbors | 2.7389× | 2.4566× | 2.6980× | 2.2104× |
| Breast Cancer | 151 | indices | 1.3202× | 1.3724× | 1.4313× | 1.4010× |
| Breast Cancer | 151 | predict_uniform | 2.9212× | 2.7750× | 2.7738× | 2.3720× |
| Breast Cancer | 151 | proba_uniform | 1.7173× | 1.6174× | 1.9585× | 1.9498× |
| Breast Cancer | 151 | regress_uniform | 1.3250× | 1.3479× | 1.2931× | 1.3410× |
| Breast Cancer | 151 | predict_distance | 1.4223× | 1.5085× | 1.4283× | 1.4681× |
| Breast Cancer | 151 | proba_distance | 1.3137× | 1.3331× | 1.3053× | 1.3436× |
| Breast Cancer | 151 | regress_distance | 3.0121× | 2.6696× | 2.6266× | 2.0751× |
| Breast Cancer | 251 | fit_query | 2.7363× | 2.0648× | 2.6635× | 2.3517× |
| Breast Cancer | 251 | kneighbors | 1.2810× | 1.3186× | 1.3245× | 1.3296× |
| Breast Cancer | 251 | indices | 1.9239× | 1.8613× | 1.6943× | 1.7501× |
| Breast Cancer | 251 | predict_uniform | 1.5539× | 1.6583× | 1.7150× | 1.7737× |
| Breast Cancer | 251 | proba_uniform | 1.4146× | 1.4878× | 1.4515× | 1.4833× |
| Breast Cancer | 251 | regress_uniform | 2.3559× | 1.8031× | 2.3868× | 1.9505× |
| Breast Cancer | 251 | predict_distance | 1.2909× | 1.2554× | 1.4697× | 1.4291× |
| Breast Cancer | 251 | proba_distance | 3.2189× | 3.1739× | 2.7815× | 2.3965× |
| Breast Cancer | 251 | regress_distance | 2.7370× | 2.4366× | 2.9551× | 2.3985× |
| Breast Cancer | 252 | fit_query | 1.4002× | 1.4925× | 1.3793× | 1.4681× |
| Breast Cancer | 252 | kneighbors | 1.6820× | 1.7350× | 1.4650× | 1.5791× |
| Breast Cancer | 252 | indices | 3.4046× | 2.5442× | 3.5859× | 3.0126× |
| Breast Cancer | 252 | predict_uniform | 1.5033× | 1.5534× | 1.5743× | 1.6050× |
| Breast Cancer | 252 | proba_uniform | 2.8485× | 2.3358× | 2.8087× | 2.5372× |
| Breast Cancer | 252 | regress_uniform | 1.4026× | 1.4113× | 1.2981× | 1.3540× |
| Breast Cancer | 252 | predict_distance | 3.0120× | 2.6720× | 2.9205× | 2.2798× |
| Breast Cancer | 252 | proba_distance | 2.9928× | 2.8679× | 3.2319× | 3.2350× |
| Breast Cancer | 252 | regress_distance | 1.3655× | 1.3676× | 1.4237× | 1.4348× |
| Offset +1e4, 4096×128 | 151 | kneighbors | 1.0187× | **0.9986×** | 1.0090× | 1.0089× |
| Offset +1e4, 4096×128 | 151 | fit_query | **0.9869×** | 1.0626× | 1.0336× | 1.0778× |
| Offset +1e4, 4096×128 | 251 | kneighbors | 1.0074× | **0.9950×** | **0.9999×** | 1.0092× |
| Offset +1e4, 4096×128 | 251 | fit_query | 1.0684× | 1.0527× | 1.0381× | 1.0313× |
| Offset +1e4, 4096×128 | 252 | kneighbors | **0.9908×** | 1.0001× | **0.9940×** | 1.0296× |
| Offset +1e4, 4096×128 | 252 | fit_query | 1.0915× | 1.0382× | 1.1222× | 1.0227× |

## Declared Digits performance screen for this protocol

All 27 Digits operations are checked against both GPU controls, both automatic replicas and both statistics: **216/216** comparisons meet the no-slowdown threshold. All pass: **True**. Distance-weighted regression meets the 1.10× development margin in **22/24** comparisons. The balanced and legacy protocols must both pass the complete screen; this report does not choose the more favorable protocol. A causal code/stage explanation is required separately.

No below-1 Digits comparisons in this protocol.

## Accuracy and source identity

All cold, warmup, probe and retained strict audits remain in the raw evidence. Every worker passes; no accuracy failure filters a timing denominator. Each repeat fixture and oracle record exactly matches the corresponding full-control fixture. Source, runtime and harness archive bytes, measured worker identities, and current source/native/harness files have been independently verified.

| Worker | Cold passes/audits | Warmup passes/audits | Probe passes/audits | Retained passes/audits |
| --- | --- | --- | --- | --- |
| Current auto | 9/9 | 225516/225516 | 60/60 | 1860/1860 |
| Current auto A/A | 9/9 | 227712/227712 | 60/60 | 1860/1860 |
| Current forced GPU | 9/9 | 170068/170068 | 60/60 | 1860/1860 |
| Immediate prior auto | 9/9 | 220595/220595 | 60/60 | 1860/1860 |
| Historical forced GPU | 9/9 | 170034/170034 | 60/60 | 1860/1860 |

All raw paired ratios, actual routing, first/middle/last summaries, position and predecessor diagnostics remain in the independent audit. They diagnose timing sensitivity without claiming an identified clock mechanism.

## Evidence

[Every raw observation](../benchmarks/results/knn-digits-final-2026-10-30-legacy-control.json) — SHA-256 `d75fef149c2112dd5ef54a4728c070f45661a0785aa4ef5c66faa2b376d6ae12`.

[Independent audit and complete crossmatching](../benchmarks/results/knn-digits-final-2026-10-30-legacy-control-audit.json).

[Full balanced control](../benchmarks/results/knn-digits-final-2026-10-30-control.json).
