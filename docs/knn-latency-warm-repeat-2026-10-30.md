# Focused continuous-warmup repeat — 2026-10-30

The repeat preserves the large Digits improvement over preceding automatic dispatch, but it does not establish universal superiority to the historical GPU implementation. All numerical audits and source/runtime/fixture checks pass. No original >5% slowdown selected for retesting recurs under the same two-replica, two-statistic criterion; a different Digits operation does meet that criterion against historical GPU.

This diagnostic covers **9 fixtures, 60 operation cells, five isolated workers and 31 paired rounds** per cell. It includes every seed (151, 251, 252) of Digits and Breast Cancer, each with all nine operations, and offset +1e4 data with 4096 rows and 128 features, each with query and fresh fit-plus-query. Digits and offset were selected after the full run showed GPU-control exceptions; Breast Cancer was added after CPU exceptions versus preceding auto were identified. Selection followed observation of the full results, so this is not a new unbiased full-panel acceptance sample.

**Both runs remain evidence.** The original 48-fixture/222-cell control used 0.25 seconds of continuous warmup per worker/operation; this repeat uses 1 second. Source, native binaries, inputs, exact ordering/distance/prediction oracles and retained pair count are unchanged. All five workers run serially in interleaved order. The warmup change prevents direct comparison of absolute latency across runs; speed ratios below compare variants within their own run. We do not replace the full run or choose the better measurement.

Current auto and current auto A/A are identical source in independent workers. Current forced GPU uses the same current source with GPU forced. Preceding auto restores `knn-auto-final-2026-10-30.zip` and its exact ABI-matched native runtime. Historical forced GPU uses `knn-small-final-2026-10-30.zip`.

All ratios are comparator/current time: values below 1× mean the automatic replica is slower and are bold. **M/P** means ratio of medians / median of corresponding paired ratios. A >5% runtime slowdown is ratio <1/1.05 (approximately 0.95238), required in both M and P for both independent automatic replicas. This descriptive recurrence check is not a confidence interval or significance test; the audit also retains the alternative ratio <0.95 threshold.

## Material findings

Digits beats preceding auto in all 27 cells for both replicas: current paired ratios **1.8418–2.5339×**, A/A **1.8060–2.4631×**. Current ratio-of-medians spans 1.8039–2.5405×; A/A spans 1.7803–2.4289×. This repeats the full-control gain.

The original Digits seed252 `proba_distance` miss against historical GPU does not meet the >5% criterion in this repeat (current M/P 0.9626/1.0160×; A/A 0.9736/1.0574×). A new seed252 `regress_distance` miss does: current M/P **0.8988/0.8843×**, A/A **0.8866/0.8823×**. Historical GPU is therefore still faster on some measurements. The new miss is against historical GPU, not preceding auto or current forced GPU.

None of the six selected Breast Cancer misses versus preceding auto repeats in both statistics and both replicas. The primary automatic worker is faster in 24/27 Breast Cancer cells by each statistic, while A/A is faster in 14/27; A/A still has individual >5% slower observations. Thirteen original >5% prior-auto misses on other small CPU fixtures were not retested and remain part of the full-run evidence.

Offset seed252 fit-plus-query retains slower observations. In the repeat, preceding-auto/current paired ratios are 0.8693× and 0.9067×, but the corresponding ratio-of-medians values are 0.9372× and 0.9976×, so they do not both cross the >5% threshold. The original current-GPU/historical-GPU misses also do not recur under the full criterion. This mismatch between statistics and replicas limits any zero-overhead claim.

## Original misses and new repeat misses

| Comparator | Full-run misses | Selected full misses | Recurring | Not recurring | New in repeat | Original misses not retested |
| --- | --- | --- | --- | --- | --- | --- |
| Preceding auto | 19 | 6 | 0 | 6 | 0 | 13 |
| Current forced GPU | 1 | 1 | 0 | 1 | 0 | 0 |
| Historical forced GPU | 2 | 2 | 0 | 2 | 1 | 0 |

| Comparator | Classification | Family | Seed | Operation | Full current M/P | Full A/A M/P | Repeat current M/P | Repeat A/A M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Preceding auto | not recurring | Breast Cancer | 151 | regress_uniform | **0.9200×** / **0.9106×** | **0.8648×** / **0.8597×** | 1.0389× / 1.0094× | **0.9450×** / **0.9475×** |
| Preceding auto | not recurring | Breast Cancer | 251 | indices | **0.9391×** / **0.9427×** | **0.9320×** / **0.9315×** | 1.1605× / 1.1563× | 1.0488× / 1.0492× |
| Preceding auto | not recurring | Breast Cancer | 251 | proba_uniform | **0.9033×** / **0.8962×** | **0.8627×** / **0.8599×** | 1.0379× / 1.0527× | **0.9724×** / **0.9867×** |
| Preceding auto | not recurring | Breast Cancer | 252 | kneighbors | **0.9341×** / **0.9302×** | **0.9142×** / **0.9010×** | 1.1370× / 1.1436× | 1.0818× / 1.0743× |
| Preceding auto | not recurring | Breast Cancer | 252 | predict_uniform | **0.8622×** / **0.8661×** | **0.8534×** / **0.8407×** | 1.0638× / 1.0638× | **0.9858×** / 1.0144× |
| Preceding auto | not recurring | Breast Cancer | 252 | regress_distance | **0.8468×** / **0.8304×** | **0.8301×** / **0.8309×** | 1.0152× / 1.0160× | **0.9223×** / **0.9351×** |
| Current forced GPU | not recurring | Offset +1e4, 4096×128 | 252 | fit_query | **0.8879×** / **0.9452×** | **0.9275×** / **0.9337×** | **0.9369×** / **0.9602×** | **0.9972×** / **0.9796×** |
| Historical forced GPU | not recurring | Offset +1e4, 4096×128 | 252 | fit_query | **0.8979×** / **0.8269×** | **0.9380×** / **0.9169×** | **0.9631×** / **0.9799×** | 1.0252× / 1.0040× |
| Historical forced GPU | not recurring | Digits | 252 | proba_distance | **0.8566×** / **0.8819×** | **0.9409×** / **0.9286×** | **0.9626×** / 1.0160× | **0.9736×** / 1.0574× |
| Historical forced GPU | new in repeat | Digits | 252 | regress_distance | 2.5239× / 2.4990× | 2.1111× / 1.9949× | **0.8988×** / **0.8843×** | **0.8866×** / **0.8823×** |

The following original prior-auto misses are outside this repeat. They are retained rather than presumed resolved.

| Original case | Operation | Full current M/P | Full A/A M/P |
| --- | --- | --- | --- |
| `real_iris_raw_k5_seed151` | proba_distance | **0.8815×** / **0.8792×** | **0.9004×** / **0.8278×** |
| `real_iris_raw_k5_seed251` | kneighbors | **0.8986×** / **0.8587×** | **0.8946×** / **0.8847×** |
| `real_iris_raw_k5_seed251` | predict_distance | **0.9427×** / **0.9377×** | **0.9289×** / **0.9008×** |
| `real_iris_raw_k5_seed252` | fit_query | **0.9508×** / **0.9480×** | **0.9284×** / **0.8963×** |
| `real_iris_raw_k5_seed252` | regress_uniform | **0.9220×** / **0.9238×** | **0.8527×** / **0.8519×** |
| `real_wine_raw_k5_seed251` | proba_uniform | **0.9269×** / **0.8840×** | **0.9197×** / **0.9320×** |
| `real_wine_raw_k5_seed251` | regress_distance | **0.9083×** / **0.9090×** | **0.8971×** / **0.8828×** |
| `real_wine_raw_k5_seed252` | kneighbors | **0.9066×** / **0.9331×** | **0.9262×** / **0.9392×** |
| `real_wine_raw_k5_seed252` | proba_distance | **0.9218×** / **0.9015×** | **0.9232×** / **0.8800×** |
| `shape_single_32_k5_seed50` | kneighbors | **0.9460×** / **0.9523×** | **0.9512×** / **0.9499×** |
| `shape_single_8_k5_seed41` | proba_uniform | **0.9186×** / **0.9009×** | **0.9164×** / **0.9371×** |
| `shape_single_8_k5_seed42` | indices | **0.8852×** / **0.8972×** | **0.9059×** / **0.8923×** |
| `shape_single_8_k5_seed42` | proba_distance | **0.8651×** / **0.8854×** | **0.8635×** / **0.8918×** |

## Every repeat timing cell

All 60 retained operation cells appear below. Times are median milliseconds from this repeat only. A/A ratios compare current auto A/A / current auto; this column exposes process variation rather than a different implementation.

| Family | Seed | Operation | Current auto ms | Current auto A/A ms | Current forced GPU ms | Preceding auto ms | Historical forced GPU ms | A/A M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 0.553583 | 0.572250 | 1.040167 | 1.037208 | 0.725583 | 1.0337× / 1.0134× |
| Digits | 151 | kneighbors | 0.359042 | 0.421083 | 0.976292 | 0.877792 | 0.886916 | 1.1728× / 1.1502× |
| Digits | 151 | indices | 0.428291 | 0.365792 | 0.572709 | 0.888459 | 0.944666 | **0.8541×** / **0.8568×** |
| Digits | 151 | predict_uniform | 0.429708 | 0.431917 | 0.441000 | 0.891125 | 0.879708 | 1.0051× / 1.0474× |
| Digits | 151 | proba_uniform | 0.436916 | 0.436000 | 0.944084 | 0.892292 | 0.484916 | **0.9979×** / **0.9798×** |
| Digits | 151 | regress_uniform | 0.435291 | 0.443083 | 0.585042 | 0.891958 | 0.418667 | 1.0179× / 1.0080× |
| Digits | 151 | predict_distance | 0.471500 | 0.489583 | 0.950208 | 0.933875 | 1.044500 | 1.0384× / 1.1065× |
| Digits | 151 | proba_distance | 0.454292 | 0.427417 | 0.526167 | 0.912208 | 0.875708 | **0.9408×** / **0.9422×** |
| Digits | 151 | regress_distance | 0.475542 | 0.465459 | 0.483084 | 0.920208 | 0.605458 | **0.9788×** / **0.9450×** |
| Digits | 251 | fit_query | 0.518834 | 0.588208 | 1.136666 | 1.047209 | 1.096834 | 1.1337× / 1.1171× |
| Digits | 251 | kneighbors | 0.443458 | 0.369833 | 1.000375 | 0.887292 | 1.031750 | **0.8340×** / **0.8319×** |
| Digits | 251 | indices | 0.430792 | 0.447583 | 0.452125 | 0.890333 | 0.941125 | 1.0390× / **0.9957×** |
| Digits | 251 | predict_uniform | 0.465875 | 0.473292 | 1.304500 | 0.910334 | 1.123834 | 1.0159× / 1.0002× |
| Digits | 251 | proba_uniform | 0.541584 | 0.523500 | 1.443000 | 1.042041 | 1.308167 | **0.9666×** / **0.9904×** |
| Digits | 251 | regress_uniform | 0.349667 | 0.443625 | 0.938459 | 0.888333 | 0.890500 | 1.2687× / 1.2504× |
| Digits | 251 | predict_distance | 0.470375 | 0.397708 | 0.982583 | 0.926292 | 1.022583 | **0.8455×** / **0.9123×** |
| Digits | 251 | proba_distance | 0.474375 | 0.468542 | 0.454458 | 0.937292 | 0.529958 | **0.9877×** / **0.9774×** |
| Digits | 251 | regress_distance | 0.480708 | 0.479334 | 1.044375 | 0.917459 | 0.899917 | **0.9971×** / 1.0102× |
| Digits | 252 | fit_query | 0.580500 | 0.533125 | 1.200166 | 1.047167 | 1.366709 | **0.9184×** / **0.9357×** |
| Digits | 252 | kneighbors | 0.441000 | 0.427709 | 0.768000 | 0.914416 | 0.976334 | **0.9699×** / **0.9641×** |
| Digits | 252 | indices | 0.412666 | 0.430333 | 0.990042 | 0.890208 | 0.958209 | 1.0428× / 1.0062× |
| Digits | 252 | predict_uniform | 0.449208 | 0.431334 | 0.931166 | 0.924709 | 0.414916 | **0.9602×** / **0.9783×** |
| Digits | 252 | proba_uniform | 0.436083 | 0.465875 | 1.007333 | 0.939750 | 0.463833 | 1.0683× / 1.0815× |
| Digits | 252 | regress_uniform | 0.450042 | 0.387250 | 0.967708 | 0.914250 | 0.998958 | **0.8605×** / **0.8829×** |
| Digits | 252 | predict_distance | 0.453292 | 0.436250 | 0.430958 | 0.942042 | 0.876083 | **0.9624×** / **0.9538×** |
| Digits | 252 | proba_distance | 0.469291 | 0.464000 | 0.997750 | 0.928250 | 0.451750 | **0.9887×** / **0.9971×** |
| Digits | 252 | regress_distance | 0.477583 | 0.484167 | 1.040208 | 0.941416 | 0.429250 | 1.0138× / **0.9967×** |
| Breast Cancer | 151 | fit_query | 0.286417 | 0.283500 | 0.324625 | 0.303917 | 0.333250 | **0.9898×** / **0.9986×** |
| Breast Cancer | 151 | kneighbors | 0.192292 | 0.195750 | 0.455625 | 0.196375 | 0.246333 | 1.0180× / 1.0161× |
| Breast Cancer | 151 | indices | 0.191083 | 0.199333 | 0.245500 | 0.199042 | 0.220333 | 1.0432× / 1.0555× |
| Breast Cancer | 151 | predict_uniform | 0.188542 | 0.198583 | 0.307208 | 0.209417 | 0.295667 | 1.0533× / 1.0458× |
| Breast Cancer | 151 | proba_uniform | 0.202167 | 0.203167 | 0.301250 | 0.210667 | 0.440209 | 1.0049× / 1.0152× |
| Breast Cancer | 151 | regress_uniform | 0.199250 | 0.219042 | 0.242333 | 0.207000 | 0.243250 | 1.0993× / 1.0290× |
| Breast Cancer | 151 | predict_distance | 0.225875 | 0.227458 | 0.352125 | 0.218500 | 0.284458 | 1.0070× / 1.0225× |
| Breast Cancer | 151 | proba_distance | 0.209042 | 0.229125 | 0.286958 | 0.224750 | 0.255292 | 1.0961× / 1.0896× |
| Breast Cancer | 151 | regress_distance | 0.205916 | 0.223625 | 0.288916 | 0.226500 | 0.273959 | 1.0860× / 1.0807× |
| Breast Cancer | 251 | fit_query | 0.290083 | 0.296250 | 0.399250 | 0.298625 | 0.372417 | 1.0213× / 1.0131× |
| Breast Cancer | 251 | kneighbors | 0.203166 | 0.216083 | 0.257000 | 0.219083 | 0.232292 | 1.0636× / 1.0704× |
| Breast Cancer | 251 | indices | 0.174459 | 0.193042 | 0.248625 | 0.202458 | 0.266584 | 1.1065× / 1.0916× |
| Breast Cancer | 251 | predict_uniform | 0.201291 | 0.205709 | 0.307667 | 0.216042 | 0.337209 | 1.0219× / 1.0325× |
| Breast Cancer | 251 | proba_uniform | 0.201125 | 0.214667 | 0.245625 | 0.208750 | 0.264542 | 1.0673× / 1.0818× |
| Breast Cancer | 251 | regress_uniform | 0.203208 | 0.213917 | 0.256125 | 0.197041 | 0.267541 | 1.0527× / 1.0835× |
| Breast Cancer | 251 | predict_distance | 0.219667 | 0.235500 | 0.273750 | 0.226417 | 0.246166 | 1.0721× / 1.0556× |
| Breast Cancer | 251 | proba_distance | 0.209125 | 0.248417 | 0.300500 | 0.242000 | 0.387667 | 1.1879× / 1.1814× |
| Breast Cancer | 251 | regress_distance | 0.223708 | 0.224125 | 0.310958 | 0.230959 | 0.563667 | 1.0019× / **0.9880×** |
| Breast Cancer | 252 | fit_query | 0.282000 | 0.289042 | 0.377042 | 0.305250 | 0.320125 | 1.0250× / **0.9948×** |
| Breast Cancer | 252 | kneighbors | 0.182125 | 0.191417 | 0.260959 | 0.207083 | 0.270334 | 1.0510× / 1.0915× |
| Breast Cancer | 252 | indices | 0.191292 | 0.187959 | 0.266833 | 0.202250 | 0.467209 | **0.9826×** / **0.9924×** |
| Breast Cancer | 252 | predict_uniform | 0.197917 | 0.213583 | 0.254458 | 0.210541 | 0.273834 | 1.0792× / 1.0500× |
| Breast Cancer | 252 | proba_uniform | 0.205791 | 0.210833 | 0.315209 | 0.200500 | 0.320375 | 1.0245× / 1.0048× |
| Breast Cancer | 252 | regress_uniform | 0.201041 | 0.213208 | 0.251917 | 0.212042 | 0.233292 | 1.0605× / 1.0491× |
| Breast Cancer | 252 | predict_distance | 0.197542 | 0.226709 | 0.287167 | 0.222291 | 0.305000 | 1.1476× / 1.1566× |
| Breast Cancer | 252 | proba_distance | 0.226083 | 0.221333 | 0.329708 | 0.228834 | 0.429375 | **0.9790×** / **0.9543×** |
| Breast Cancer | 252 | regress_distance | 0.211375 | 0.232667 | 0.259417 | 0.214584 | 0.260542 | 1.1007× / 1.0673× |
| Offset +1e4, 4096×128 | 151 | kneighbors | 0.395666 | 0.402208 | 0.390416 | 0.386625 | 0.388958 | 1.0165× / 1.0206× |
| Offset +1e4, 4096×128 | 151 | fit_query | 0.660041 | 0.674500 | 0.687083 | 0.660417 | 0.651541 | 1.0219× / **0.9571×** |
| Offset +1e4, 4096×128 | 251 | kneighbors | 0.407292 | 0.388375 | 0.408792 | 0.385208 | 0.375125 | **0.9536×** / **0.9761×** |
| Offset +1e4, 4096×128 | 251 | fit_query | 0.738500 | 0.730042 | 0.697375 | 0.715875 | 0.765584 | **0.9885×** / 1.0027× |
| Offset +1e4, 4096×128 | 252 | kneighbors | 0.398541 | 0.418667 | 0.385792 | 0.406583 | 0.388125 | 1.0505× / 1.0000× |
| Offset +1e4, 4096×128 | 252 | fit_query | 0.743625 | 0.698625 | 0.696667 | 0.696916 | 0.716208 | **0.9395×** / **0.9981×** |

## Every repeat ratio against Preceding auto

Each row retains both ratio statistics and both automatic replicas. Bold values are slower observations, even when they do not meet the >5% cross-replica criterion.

| Family | Seed | Operation | Current ratio of medians | Current median paired | A/A ratio of medians | A/A median paired |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.8736× | 1.8636× | 1.8125× | 1.8060× |
| Digits | 151 | kneighbors | 2.4448× | 2.4882× | 2.0846× | 2.0850× |
| Digits | 151 | indices | 2.0744× | 2.1481× | 2.4289× | 2.3955× |
| Digits | 151 | predict_uniform | 2.0738× | 2.0283× | 2.0632× | 2.0870× |
| Digits | 151 | proba_uniform | 2.0423× | 2.0452× | 2.0465× | 2.0702× |
| Digits | 151 | regress_uniform | 2.0491× | 2.0491× | 2.0131× | 2.0369× |
| Digits | 151 | predict_distance | 1.9806× | 2.0299× | 1.9075× | 1.8901× |
| Digits | 151 | proba_distance | 2.0080× | 1.9851× | 2.1342× | 2.1324× |
| Digits | 151 | regress_distance | 1.9351× | 1.9369× | 1.9770× | 1.9811× |
| Digits | 251 | fit_query | 2.0184× | 2.0540× | 1.7803× | 1.8078× |
| Digits | 251 | kneighbors | 2.0008× | 2.0514× | 2.3992× | 2.4631× |
| Digits | 251 | indices | 2.0667× | 2.0674× | 1.9892× | 2.0034× |
| Digits | 251 | predict_uniform | 1.9540× | 2.0230× | 1.9234× | 1.9907× |
| Digits | 251 | proba_uniform | 1.9241× | 1.9169× | 1.9905× | 1.9549× |
| Digits | 251 | regress_uniform | 2.5405× | 2.5339× | 2.0024× | 2.0161× |
| Digits | 251 | predict_distance | 1.9693× | 2.0093× | 2.3291× | 2.2814× |
| Digits | 251 | proba_distance | 1.9758× | 1.9708× | 2.0004× | 2.0431× |
| Digits | 251 | regress_distance | 1.9086× | 1.9280× | 1.9140× | 1.9485× |
| Digits | 252 | fit_query | 1.8039× | 1.8418× | 1.9642× | 1.9684× |
| Digits | 252 | kneighbors | 2.0735× | 2.0900× | 2.1379× | 2.1660× |
| Digits | 252 | indices | 2.1572× | 2.1472× | 2.0686× | 2.0666× |
| Digits | 252 | predict_uniform | 2.0585× | 2.0374× | 2.1438× | 2.1073× |
| Digits | 252 | proba_uniform | 2.1550× | 2.1483× | 2.0172× | 1.9900× |
| Digits | 252 | regress_uniform | 2.0315× | 2.0165× | 2.3609× | 2.3760× |
| Digits | 252 | predict_distance | 2.0782× | 2.1135× | 2.1594× | 2.1544× |
| Digits | 252 | proba_distance | 1.9780× | 2.0057× | 2.0005× | 2.0187× |
| Digits | 252 | regress_distance | 1.9712× | 1.9564× | 1.9444× | 1.9602× |
| Breast Cancer | 151 | fit_query | 1.0611× | 1.0703× | 1.0720× | 1.0765× |
| Breast Cancer | 151 | kneighbors | 1.0212× | 1.0053× | 1.0032× | **0.9483×** |
| Breast Cancer | 151 | indices | 1.0417× | 1.0613× | **0.9985×** | 1.0150× |
| Breast Cancer | 151 | predict_uniform | 1.1107× | 1.1306× | 1.0546× | 1.0813× |
| Breast Cancer | 151 | proba_uniform | 1.0420× | 1.0164× | 1.0369× | 1.0344× |
| Breast Cancer | 151 | regress_uniform | 1.0389× | 1.0094× | **0.9450×** | **0.9475×** |
| Breast Cancer | 151 | predict_distance | **0.9673×** | **0.9714×** | **0.9606×** | **0.9482×** |
| Breast Cancer | 151 | proba_distance | 1.0751× | 1.0384× | **0.9809×** | **0.9574×** |
| Breast Cancer | 151 | regress_distance | 1.1000× | 1.1147× | 1.0129× | **0.9753×** |
| Breast Cancer | 251 | fit_query | 1.0294× | 1.0119× | 1.0080× | **0.9989×** |
| Breast Cancer | 251 | kneighbors | 1.0783× | 1.0586× | 1.0139× | 1.0122× |
| Breast Cancer | 251 | indices | 1.1605× | 1.1563× | 1.0488× | 1.0492× |
| Breast Cancer | 251 | predict_uniform | 1.0733× | 1.0641× | 1.0502× | 1.0323× |
| Breast Cancer | 251 | proba_uniform | 1.0379× | 1.0527× | **0.9724×** | **0.9867×** |
| Breast Cancer | 251 | regress_uniform | **0.9697×** | **0.9670×** | **0.9211×** | **0.9020×** |
| Breast Cancer | 251 | predict_distance | 1.0307× | 1.0289× | **0.9614×** | **0.9966×** |
| Breast Cancer | 251 | proba_distance | 1.1572× | 1.1490× | **0.9742×** | **0.9943×** |
| Breast Cancer | 251 | regress_distance | 1.0324× | 1.0505× | 1.0305× | 1.0223× |
| Breast Cancer | 252 | fit_query | 1.0824× | 1.0619× | 1.0561× | 1.0315× |
| Breast Cancer | 252 | kneighbors | 1.1370× | 1.1436× | 1.0818× | 1.0743× |
| Breast Cancer | 252 | indices | 1.0573× | 1.0709× | 1.0760× | 1.0819× |
| Breast Cancer | 252 | predict_uniform | 1.0638× | 1.0638× | **0.9858×** | 1.0144× |
| Breast Cancer | 252 | proba_uniform | **0.9743×** | **0.9717×** | **0.9510×** | **0.9459×** |
| Breast Cancer | 252 | regress_uniform | 1.0547× | 1.0301× | **0.9945×** | 1.0045× |
| Breast Cancer | 252 | predict_distance | 1.1253× | 1.1496× | **0.9805×** | **0.9904×** |
| Breast Cancer | 252 | proba_distance | 1.0122× | 1.0395× | 1.0339× | 1.0584× |
| Breast Cancer | 252 | regress_distance | 1.0152× | 1.0160× | **0.9223×** | **0.9351×** |
| Offset +1e4, 4096×128 | 151 | kneighbors | **0.9771×** | **0.9985×** | **0.9613×** | **0.9888×** |
| Offset +1e4, 4096×128 | 151 | fit_query | 1.0006× | 1.0053× | **0.9791×** | 1.0271× |
| Offset +1e4, 4096×128 | 251 | kneighbors | **0.9458×** | **0.9703×** | **0.9918×** | 1.0047× |
| Offset +1e4, 4096×128 | 251 | fit_query | **0.9694×** | 1.0123× | **0.9806×** | 1.0368× |
| Offset +1e4, 4096×128 | 252 | kneighbors | 1.0202× | 1.0320× | **0.9711×** | 1.0553× |
| Offset +1e4, 4096×128 | 252 | fit_query | **0.9372×** | **0.8693×** | **0.9976×** | **0.9067×** |

## Every repeat ratio against Current forced GPU

Each row retains both ratio statistics and both automatic replicas. Bold values are slower observations, even when they do not meet the >5% cross-replica criterion.

| Family | Seed | Operation | Current ratio of medians | Current median paired | A/A ratio of medians | A/A median paired |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.8790× | 1.9180× | 1.8177× | 1.8784× |
| Digits | 151 | kneighbors | 2.7192× | 2.6022× | 2.3185× | 2.2804× |
| Digits | 151 | indices | 1.3372× | 1.5623× | 1.5657× | 1.7212× |
| Digits | 151 | predict_uniform | 1.0263× | 1.1406× | 1.0210× | **0.9966×** |
| Digits | 151 | proba_uniform | 2.1608× | 2.2183× | 2.1653× | 2.2890× |
| Digits | 151 | regress_uniform | 1.3440× | 1.4443× | 1.3204× | 1.4753× |
| Digits | 151 | predict_distance | 2.0153× | 2.2242× | 1.9409× | 1.9486× |
| Digits | 151 | proba_distance | 1.1582× | 1.2175× | 1.2310× | 1.2172× |
| Digits | 151 | regress_distance | 1.0159× | 1.0625× | 1.0379× | 1.1344× |
| Digits | 251 | fit_query | 2.1908× | 2.2516× | 1.9324× | 2.0067× |
| Digits | 251 | kneighbors | 2.2559× | 2.3845× | 2.7049× | 2.6467× |
| Digits | 251 | indices | 1.0495× | 1.1240× | 1.0101× | 1.1023× |
| Digits | 251 | predict_uniform | 2.8001× | 2.7842× | 2.7562× | 2.6621× |
| Digits | 251 | proba_uniform | 2.6644× | 2.6479× | 2.7564× | 2.7158× |
| Digits | 251 | regress_uniform | 2.6839× | 2.6444× | 2.1154× | 2.0262× |
| Digits | 251 | predict_distance | 2.0889× | 2.1125× | 2.4706× | 2.2160× |
| Digits | 251 | proba_distance | **0.9580×** | 1.0348× | **0.9699×** | 1.0400× |
| Digits | 251 | regress_distance | 2.1726× | 2.1237× | 2.1788× | 2.1759× |
| Digits | 252 | fit_query | 2.0675× | 2.1394× | 2.2512× | 2.2926× |
| Digits | 252 | kneighbors | 1.7415× | 1.6788× | 1.7956× | 1.8580× |
| Digits | 252 | indices | 2.3991× | 2.3991× | 2.3006× | 2.4524× |
| Digits | 252 | predict_uniform | 2.0729× | 2.2784× | 2.1588× | 2.0474× |
| Digits | 252 | proba_uniform | 2.3100× | 2.2608× | 2.1622× | 2.1563× |
| Digits | 252 | regress_uniform | 2.1503× | 2.2132× | 2.4989× | 2.4899× |
| Digits | 252 | predict_distance | **0.9507×** | **0.9648×** | **0.9879×** | 1.0114× |
| Digits | 252 | proba_distance | 2.1261× | 2.0773× | 2.1503× | 2.0758× |
| Digits | 252 | regress_distance | 2.1781× | 2.1781× | 2.1484× | 2.1831× |
| Breast Cancer | 151 | fit_query | 1.1334× | 1.1553× | 1.1451× | 1.1664× |
| Breast Cancer | 151 | kneighbors | 2.3694× | 2.0183× | 2.3276× | 1.9772× |
| Breast Cancer | 151 | indices | 1.2848× | 1.3180× | 1.2316× | 1.2417× |
| Breast Cancer | 151 | predict_uniform | 1.6294× | 1.6370× | 1.5470× | 1.5701× |
| Breast Cancer | 151 | proba_uniform | 1.4901× | 1.5896× | 1.4828× | 1.5680× |
| Breast Cancer | 151 | regress_uniform | 1.2162× | 1.2505× | 1.1063× | 1.1385× |
| Breast Cancer | 151 | predict_distance | 1.5589× | 1.5393× | 1.5481× | 1.5773× |
| Breast Cancer | 151 | proba_distance | 1.3727× | 1.3646× | 1.2524× | 1.2578× |
| Breast Cancer | 151 | regress_distance | 1.4031× | 1.4399× | 1.2920× | 1.4218× |
| Breast Cancer | 251 | fit_query | 1.3763× | 1.4248× | 1.3477× | 1.4798× |
| Breast Cancer | 251 | kneighbors | 1.2650× | 1.2947× | 1.1894× | 1.2103× |
| Breast Cancer | 251 | indices | 1.4251× | 1.4568× | 1.2879× | 1.3239× |
| Breast Cancer | 251 | predict_uniform | 1.5285× | 1.5098× | 1.4956× | 1.5762× |
| Breast Cancer | 251 | proba_uniform | 1.2213× | 1.2187× | 1.1442× | 1.1342× |
| Breast Cancer | 251 | regress_uniform | 1.2604× | 1.3106× | 1.1973× | 1.2160× |
| Breast Cancer | 251 | predict_distance | 1.2462× | 1.2423× | 1.1624× | 1.2401× |
| Breast Cancer | 251 | proba_distance | 1.4369× | 1.4624× | 1.2097× | 1.2051× |
| Breast Cancer | 251 | regress_distance | 1.3900× | 1.3313× | 1.3874× | 1.4067× |
| Breast Cancer | 252 | fit_query | 1.3370× | 1.2854× | 1.3045× | 1.3120× |
| Breast Cancer | 252 | kneighbors | 1.4329× | 1.5032× | 1.3633× | 1.3698× |
| Breast Cancer | 252 | indices | 1.3949× | 1.3371× | 1.4196× | 1.4002× |
| Breast Cancer | 252 | predict_uniform | 1.2857× | 1.2315× | 1.1914× | 1.1708× |
| Breast Cancer | 252 | proba_uniform | 1.5317× | 1.5400× | 1.4951× | 1.4348× |
| Breast Cancer | 252 | regress_uniform | 1.2531× | 1.2714× | 1.1816× | 1.1980× |
| Breast Cancer | 252 | predict_distance | 1.4537× | 1.4659× | 1.2667× | 1.2493× |
| Breast Cancer | 252 | proba_distance | 1.4583× | 1.5036× | 1.4896× | 1.4832× |
| Breast Cancer | 252 | regress_distance | 1.2273× | 1.2102× | 1.1150× | 1.1105× |
| Offset +1e4, 4096×128 | 151 | kneighbors | **0.9867×** | 1.0202× | **0.9707×** | **0.9707×** |
| Offset +1e4, 4096×128 | 151 | fit_query | 1.0410× | **0.9847×** | 1.0187× | 1.0376× |
| Offset +1e4, 4096×128 | 251 | kneighbors | 1.0037× | **0.9878×** | 1.0526× | 1.0108× |
| Offset +1e4, 4096×128 | 251 | fit_query | **0.9443×** | **0.9516×** | **0.9553×** | **0.9977×** |
| Offset +1e4, 4096×128 | 252 | kneighbors | **0.9680×** | **0.9453×** | **0.9215×** | **0.9524×** |
| Offset +1e4, 4096×128 | 252 | fit_query | **0.9369×** | **0.9602×** | **0.9972×** | **0.9796×** |

## Every repeat ratio against Historical forced GPU

Each row retains both ratio statistics and both automatic replicas. Bold values are slower observations, even when they do not meet the >5% cross-replica criterion.

| Family | Seed | Operation | Current ratio of medians | Current median paired | A/A ratio of medians | A/A median paired |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.3107× | 1.2287× | 1.2679× | 1.3044× |
| Digits | 151 | kneighbors | 2.4702× | 2.5037× | 2.1063× | 2.1368× |
| Digits | 151 | indices | 2.2057× | 2.2139× | 2.5825× | 2.0563× |
| Digits | 151 | predict_uniform | 2.0472× | 1.9064× | 2.0368× | 1.8589× |
| Digits | 151 | proba_uniform | 1.1099× | 1.2000× | 1.1122× | 1.2166× |
| Digits | 151 | regress_uniform | **0.9618×** | 1.0820× | **0.9449×** | **0.9839×** |
| Digits | 151 | predict_distance | 2.2153× | 2.1798× | 2.1334× | 2.1827× |
| Digits | 151 | proba_distance | 1.9276× | 2.0064× | 2.0488× | 2.0687× |
| Digits | 151 | regress_distance | 1.2732× | 1.2905× | 1.3008× | 1.3008× |
| Digits | 251 | fit_query | 2.1140× | 2.3208× | 1.8647× | 2.0989× |
| Digits | 251 | kneighbors | 2.3266× | 2.3280× | 2.7898× | 2.6674× |
| Digits | 251 | indices | 2.1846× | 2.1573× | 2.1027× | 2.1495× |
| Digits | 251 | predict_uniform | 2.4123× | 2.6815× | 2.3745× | 2.5032× |
| Digits | 251 | proba_uniform | 2.4154× | 2.4756× | 2.4989× | 2.5132× |
| Digits | 251 | regress_uniform | 2.5467× | 2.4755× | 2.0073× | 2.0073× |
| Digits | 251 | predict_distance | 2.1740× | 2.3007× | 2.5712× | 2.4659× |
| Digits | 251 | proba_distance | 1.1172× | 1.2578× | 1.1311× | 1.1811× |
| Digits | 251 | regress_distance | 1.8721× | 1.8037× | 1.8774× | 1.8336× |
| Digits | 252 | fit_query | 2.3544× | 2.3130× | 2.5636× | 2.4595× |
| Digits | 252 | kneighbors | 2.2139× | 2.2476× | 2.2827× | 2.3083× |
| Digits | 252 | indices | 2.3220× | 2.2654× | 2.2267× | 2.3649× |
| Digits | 252 | predict_uniform | **0.9237×** | **0.9494×** | **0.9619×** | **0.9636×** |
| Digits | 252 | proba_uniform | 1.0636× | 1.1761× | **0.9956×** | 1.1381× |
| Digits | 252 | regress_uniform | 2.2197× | 2.2219× | 2.5796× | 2.5769× |
| Digits | 252 | predict_distance | 1.9327× | 1.8979× | 2.0082× | 1.9513× |
| Digits | 252 | proba_distance | **0.9626×** | 1.0160× | **0.9736×** | 1.0574× |
| Digits | 252 | regress_distance | **0.8988×** | **0.8843×** | **0.8866×** | **0.8823×** |
| Breast Cancer | 151 | fit_query | 1.1635× | 1.2146× | 1.1755× | 1.2064× |
| Breast Cancer | 151 | kneighbors | 1.2810× | 1.3088× | 1.2584× | 1.2871× |
| Breast Cancer | 151 | indices | 1.1531× | 1.2178× | 1.1054× | 1.0694× |
| Breast Cancer | 151 | predict_uniform | 1.5682× | 1.6087× | 1.4889× | 1.6243× |
| Breast Cancer | 151 | proba_uniform | 2.1775× | 2.2928× | 2.1667× | 2.2464× |
| Breast Cancer | 151 | regress_uniform | 1.2208× | 1.2208× | 1.1105× | 1.1023× |
| Breast Cancer | 151 | predict_distance | 1.2594× | 1.3879× | 1.2506× | 1.3226× |
| Breast Cancer | 151 | proba_distance | 1.2212× | 1.2134× | 1.1142× | 1.0837× |
| Breast Cancer | 151 | regress_distance | 1.3304× | 1.3166× | 1.2251× | 1.2969× |
| Breast Cancer | 251 | fit_query | 1.2838× | 1.2582× | 1.2571× | 1.2841× |
| Breast Cancer | 251 | kneighbors | 1.1434× | 1.1417× | 1.0750× | 1.0696× |
| Breast Cancer | 251 | indices | 1.5281× | 1.4739× | 1.3810× | 1.3598× |
| Breast Cancer | 251 | predict_uniform | 1.6752× | 1.7582× | 1.6393× | 1.7694× |
| Breast Cancer | 251 | proba_uniform | 1.3153× | 1.3244× | 1.2323× | 1.2442× |
| Breast Cancer | 251 | regress_uniform | 1.3166× | 1.4059× | 1.2507× | 1.2352× |
| Breast Cancer | 251 | predict_distance | 1.1206× | 1.1017× | 1.0453× | 1.0132× |
| Breast Cancer | 251 | proba_distance | 1.8538× | 1.8144× | 1.5605× | 1.1780× |
| Breast Cancer | 251 | regress_distance | 2.5197× | 2.2202× | 2.5150× | 2.4332× |
| Breast Cancer | 252 | fit_query | 1.1352× | 1.1533× | 1.1075× | 1.0950× |
| Breast Cancer | 252 | kneighbors | 1.4843× | 1.5099× | 1.4123× | 1.5187× |
| Breast Cancer | 252 | indices | 2.4424× | 2.1676× | 2.4857× | 2.6049× |
| Breast Cancer | 252 | predict_uniform | 1.3836× | 1.3950× | 1.2821× | 1.2625× |
| Breast Cancer | 252 | proba_uniform | 1.5568× | 1.5349× | 1.5196× | 1.5259× |
| Breast Cancer | 252 | regress_uniform | 1.1604× | 1.1122× | 1.0942× | 1.1127× |
| Breast Cancer | 252 | predict_distance | 1.5440× | 1.5286× | 1.3453× | 1.2950× |
| Breast Cancer | 252 | proba_distance | 1.8992× | 1.6568× | 1.9400× | 1.9677× |
| Breast Cancer | 252 | regress_distance | 1.2326× | 1.2222× | 1.1198× | 1.1284× |
| Offset +1e4, 4096×128 | 151 | kneighbors | **0.9830×** | 1.0320× | **0.9671×** | **0.9765×** |
| Offset +1e4, 4096×128 | 151 | fit_query | **0.9871×** | **0.9771×** | **0.9660×** | 1.0235× |
| Offset +1e4, 4096×128 | 251 | kneighbors | **0.9210×** | **0.9700×** | **0.9659×** | **0.9779×** |
| Offset +1e4, 4096×128 | 251 | fit_query | 1.0367× | **0.9862×** | 1.0487× | 1.0047× |
| Offset +1e4, 4096×128 | 252 | kneighbors | **0.9739×** | **0.9745×** | **0.9270×** | **0.9978×** |
| Offset +1e4, 4096×128 | 252 | fit_query | **0.9631×** | **0.9799×** | 1.0252× | 1.0040× |

## Accuracy and integrity

Every cold, warmup, probe and retained strict audit passes for all five workers. The JSON retains each observation and actual query/fit-backend telemetry; no failed result was filtered out of timing denominators. Counts below show strict passes / audits.

| Worker | Cold | Warmup | Probe | Retained |
| --- | --- | --- | --- | --- |
| Current auto | 9/9 | 182121/182121 | 60/60 | 1860/1860 |
| Current auto A/A | 9/9 | 175096/175096 | 60/60 | 1860/1860 |
| Current forced GPU | 9/9 | 164614/164614 | 60/60 | 1860/1860 |
| Preceding auto | 9/9 | 138943/138943 | 60/60 | 1860/1860 |
| Historical forced GPU | 9/9 | 167642/167642 | 60/60 | 1860/1860 |

Validation recomputes all summaries and comparisons from the repeat raw file with `validate(report, expected_cases=9)`. All nine fixture metadata and oracle inventories exactly match the original full control. Source manifests/archives, measured harness files, current on-disk source, and all four native worker source/binary hashes match the full run; prior automatic runtime restoration is independently verified. Current runtime artifact matches the measured binary.

Native binary SHA-256: `bacc58cf95a565df84eb40dd214d97c3bdf646153ff775e5488105f67120c2b7`. C++ SHA-256: `3f85be27df45b7b4012b4f19765974f670c8a2b79a7e69f1209acd861cd0ca07`.

## Evidence

[Repeat raw results](../benchmarks/results/knn-latency-final-2026-10-30-warm-exceptions-repeat.json) — SHA-256 `310495b7a788c628e821a282dba0b89396281ba09c07d7eb5360cbafceda4bcf`.

[Original full control](../benchmarks/results/knn-latency-final-2026-10-30-control.json) — SHA-256 `e12764076f2aaa5665324f2ec6a8ab2288eef9c5569bed33ba2a64434f305e95`.

[Independent repeat audit, all paired ratios, routing and cross-run flags](../benchmarks/results/knn-latency-final-2026-10-30-warm-exceptions-repeat-audit.json).
