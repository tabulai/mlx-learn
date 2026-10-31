# Predeclared Gaussian and clustered 128D preservation repeat

All **18 operation cells**, both automatic replicas and all five workers are retained. Accuracy, runtime identity and numerical-oracle validation pass. The two isolated >5% full-control flags were covered by these predeclared panels; the panel-specific crosschecks below retain both the full result and the repeat, including smaller or inconsistent slowdowns.

This is the `preservation-repeat` portion of the [saved predeclaration](../benchmarks/results/knn-startup-final2-legacy-predeclaration-2026-10-30.json) (SHA-256 `882bbe578f6a0e94c3e172f4d2bfcc7d801a63ac7a03bc985e9fb74f5ab3f131`). The original Digits/Breast/offset panel was retained, and all six Gaussian 128D seeds plus all three clustered 128D seeds were added before the final preservation results. The exact requested inventories are verified, with no seed or operation removed.

Each operation uses 31 serial interleaved rounds across current auto, identical current auto A/A, current forced GPU, immediate prior auto (`knn-latency-final-2026-10-30` plus its exact archived native runtime), and historical forced GPU (`knn-small-final-2026-10-30`). This panel uses the legacy forward/reverse ordering and 1 second of continuous warmup. The full 48-fixture control uses balanced Williams ordering and 0.25 seconds of warmup. Absolute milliseconds across these runs are not directly comparable; ratios compare implementations within their own run. Neither run replaces the other, and no faster window or run is selected.

M/P means ratio of medians / median paired ratio. Ratios are comparator/current; below 1× means the automatic replica is slower and is bold. A >5% runtime slowdown means ratio <1/1.05 for BOTH statistics and BOTH independent automatic replicas. This recurrence screen is descriptive, not a confidence interval or significance test.

## Full-control exceptions and repeat crosscheck

| Comparator | Flags in full 222 cells | Covered here | Recurring | Not recurring | New flags |
| --- | --- | --- | --- | --- | --- |
| Immediate prior auto | 1 | 0 | 0 | 0 | 0 |
| Current forced GPU | 1 | 1 | 0 | 1 | 0 |
| Historical forced GPU | 0 | 0 | 0 | 0 | 0 |

| Comparator | Classification | Family | Seed | Operation | Full current M/P | Full A/A M/P | Repeat current M/P | Repeat A/A M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Current forced GPU | not recurring | Gaussian 4096×128 | 552 | kneighbors | **0.9214×** / **0.9429×** | **0.9201×** / **0.9493×** | 1.0619× / 1.0121× | 1.0281× / 1.0263× |

## Every slower observation in either statistic

| Comparator | Replica | Family | Seed | Operation | M/P |
| --- | --- | --- | --- | --- | --- |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 151 | kneighbors | **0.9678×** / **0.9930×** |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 151 | kneighbors | 1.0029× / **0.9994×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 151 | kneighbors | **0.9589×** / **0.9899×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 151 | kneighbors | **0.9936×** / 1.0010× |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 151 | kneighbors | **0.9454×** / **0.9530×** |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 151 | kneighbors | **0.9797×** / **0.9577×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 151 | fit_query | **0.9977×** / 1.0304× |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 151 | fit_query | **0.9765×** / 1.0006× |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 251 | kneighbors | 1.0233× / **0.9732×** |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 251 | kneighbors | 1.0127× / **0.9777×** |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 251 | kneighbors | 1.0396× / **0.9949×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 251 | fit_query | **0.8967×** / 1.0377× |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 251 | fit_query | **0.9646×** / 1.1126× |
| Current forced GPU | Current auto | Gaussian 4096×128 | 251 | fit_query | **0.9562×** / **0.9343×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 251 | fit_query | **0.9490×** / 1.0461× |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 252 | kneighbors | **0.9879×** / 1.0281× |
| Current forced GPU | Current auto | Gaussian 4096×128 | 252 | fit_query | 1.0152× / **0.9789×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 252 | fit_query | **0.9695×** / **0.9723×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 351 | kneighbors | 1.0142× / **0.9750×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 351 | kneighbors | **0.9512×** / **0.9632×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 351 | kneighbors | **0.9388×** / **0.9670×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 351 | kneighbors | **0.9829×** / **0.9768×** |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 351 | kneighbors | **0.9700×** / **0.9780×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 351 | fit_query | **0.9583×** / **0.9671×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 351 | fit_query | **0.9567×** / **0.9848×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 451 | kneighbors | **0.9482×** / **0.9363×** |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 451 | kneighbors | 1.0061× / **0.9572×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 451 | kneighbors | **0.9548×** / **0.9455×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 451 | kneighbors | 1.0131× / **0.9987×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 451 | kneighbors | **0.9686×** / **0.9825×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 451 | fit_query | 1.0002× / **0.9599×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 451 | fit_query | **0.9878×** / 1.0197× |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 451 | fit_query | **0.9936×** / **0.9836×** |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 552 | kneighbors | **0.9826×** / 1.0008× |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 552 | kneighbors | **0.9754×** / **0.9706×** |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 552 | kneighbors | **0.9444×** / **0.9476×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 552 | fit_query | **0.9795×** / 1.1869× |
| Current forced GPU | Current auto | Gaussian 4096×128 | 552 | fit_query | **0.9799×** / 1.0176× |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 552 | fit_query | 1.0120× / **0.9859×** |
| Immediate prior auto | Current auto A/A | Clusters 4096×128 | 151 | kneighbors | **0.9913×** / 1.0109× |
| Current forced GPU | Current auto | Clusters 4096×128 | 151 | kneighbors | **0.9831×** / **0.9866×** |
| Current forced GPU | Current auto A/A | Clusters 4096×128 | 151 | kneighbors | **0.9672×** / **0.9983×** |
| Historical forced GPU | Current auto | Clusters 4096×128 | 151 | kneighbors | **0.9779×** / 1.0069× |
| Historical forced GPU | Current auto A/A | Clusters 4096×128 | 151 | kneighbors | **0.9620×** / **0.9658×** |
| Current forced GPU | Current auto | Clusters 4096×128 | 151 | fit_query | **0.9819×** / 1.0041× |
| Current forced GPU | Current auto | Clusters 4096×128 | 251 | kneighbors | **0.9920×** / 1.0062× |
| Historical forced GPU | Current auto | Clusters 4096×128 | 251 | kneighbors | **0.9853×** / **0.9931×** |
| Historical forced GPU | Current auto A/A | Clusters 4096×128 | 251 | kneighbors | **0.9988×** / **0.9983×** |
| Current forced GPU | Current auto A/A | Clusters 4096×128 | 251 | fit_query | **0.9973×** / **0.9910×** |
| Current forced GPU | Current auto | Clusters 4096×128 | 252 | kneighbors | **0.9602×** / **0.9507×** |
| Current forced GPU | Current auto A/A | Clusters 4096×128 | 252 | kneighbors | **0.9832×** / **0.9690×** |
| Historical forced GPU | Current auto | Clusters 4096×128 | 252 | kneighbors | 1.0002× / **0.9974×** |
| Current forced GPU | Current auto | Clusters 4096×128 | 252 | fit_query | **0.9775×** / **0.9848×** |
| Current forced GPU | Current auto A/A | Clusters 4096×128 | 252 | fit_query | 1.0039× / **0.9960×** |

## Every repeat timing cell

All times are median milliseconds from this repeat only.

| Family | Seed | Operation | Current auto ms | Current auto A/A ms | Current forced GPU ms | Immediate prior auto ms | Historical forced GPU ms | A/A over auto M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 4096×128 | 151 | kneighbors | 0.405291 | 0.391125 | 0.388625 | 0.392250 | 0.383166 | **0.9650×** / **0.9863×** |
| Gaussian 4096×128 | 151 | fit_query | 0.604959 | 0.618083 | 0.603541 | 0.638709 | 0.631459 | 1.0217× / 1.0201× |
| Gaussian 4096×128 | 251 | kneighbors | 0.383208 | 0.387209 | 0.395084 | 0.392125 | 0.402542 | 1.0104× / **0.9778×** |
| Gaussian 4096×128 | 251 | fit_query | 0.861958 | 0.801333 | 0.824208 | 0.772958 | 0.817958 | **0.9297×** / **0.9430×** |
| Gaussian 4096×128 | 252 | kneighbors | 0.393917 | 0.384375 | 0.396250 | 0.389166 | 0.403834 | **0.9758×** / **0.9868×** |
| Gaussian 4096×128 | 252 | fit_query | 0.606625 | 0.635208 | 0.615834 | 0.658042 | 0.655208 | 1.0471× / 1.0072× |
| Gaussian 4096×128 | 351 | kneighbors | 0.386833 | 0.391959 | 0.367958 | 0.392334 | 0.380209 | 1.0133× / **0.9786×** |
| Gaussian 4096×128 | 351 | fit_query | 0.667333 | 0.668417 | 0.639500 | 0.681000 | 0.674542 | 1.0016× / **0.9477×** |
| Gaussian 4096×128 | 451 | kneighbors | 0.415750 | 0.391833 | 0.396958 | 0.394209 | 0.402708 | **0.9425×** / **0.9642×** |
| Gaussian 4096×128 | 451 | fit_query | 0.722250 | 0.718000 | 0.722375 | 0.773459 | 0.713416 | **0.9941×** / 1.0110× |
| Gaussian 4096×128 | 552 | kneighbors | 0.410625 | 0.424125 | 0.436041 | 0.416750 | 0.400541 | 1.0329× / **0.9942×** |
| Gaussian 4096×128 | 552 | fit_query | 0.655625 | 0.634834 | 0.642458 | 0.642208 | 0.667208 | **0.9683×** / 1.0209× |
| Clusters 4096×128 | 151 | kneighbors | 0.414625 | 0.421459 | 0.407625 | 0.417792 | 0.405459 | 1.0165× / **0.9975×** |
| Clusters 4096×128 | 151 | fit_query | 0.552292 | 0.524916 | 0.542292 | 0.577958 | 0.568792 | **0.9504×** / **0.9882×** |
| Clusters 4096×128 | 251 | kneighbors | 0.385333 | 0.380125 | 0.382250 | 0.391417 | 0.379666 | **0.9865×** / **0.9833×** |
| Clusters 4096×128 | 251 | fit_query | 0.525833 | 0.531583 | 0.530167 | 0.604958 | 0.564416 | 1.0109× / 1.0085× |
| Clusters 4096×128 | 252 | kneighbors | 0.401125 | 0.391750 | 0.385166 | 0.403041 | 0.401208 | **0.9766×** / **0.9855×** |
| Clusters 4096×128 | 252 | fit_query | 0.553666 | 0.539125 | 0.541209 | 0.583375 | 0.579333 | **0.9737×** / **0.9780×** |

## Every cell against Immediate prior auto

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 4096×128 | 151 | kneighbors | **0.9678×** | **0.9930×** | 1.0029× | **0.9994×** |
| Gaussian 4096×128 | 151 | fit_query | 1.0558× | 1.1011× | 1.0334× | 1.0472× |
| Gaussian 4096×128 | 251 | kneighbors | 1.0233× | **0.9732×** | 1.0127× | **0.9777×** |
| Gaussian 4096×128 | 251 | fit_query | **0.8967×** | 1.0377× | **0.9646×** | 1.1126× |
| Gaussian 4096×128 | 252 | kneighbors | **0.9879×** | 1.0281× | 1.0125× | 1.0463× |
| Gaussian 4096×128 | 252 | fit_query | 1.0848× | 1.0578× | 1.0359× | 1.0609× |
| Gaussian 4096×128 | 351 | kneighbors | 1.0142× | **0.9750×** | 1.0010× | 1.0127× |
| Gaussian 4096×128 | 351 | fit_query | 1.0205× | 1.0361× | 1.0188× | 1.0573× |
| Gaussian 4096×128 | 451 | kneighbors | **0.9482×** | **0.9363×** | 1.0061× | **0.9572×** |
| Gaussian 4096×128 | 451 | fit_query | 1.0709× | 1.0879× | 1.0772× | 1.0469× |
| Gaussian 4096×128 | 552 | kneighbors | 1.0149× | 1.0054× | **0.9826×** | 1.0008× |
| Gaussian 4096×128 | 552 | fit_query | **0.9795×** | 1.1869× | 1.0116× | 1.0097× |
| Clusters 4096×128 | 151 | kneighbors | 1.0076× | 1.0149× | **0.9913×** | 1.0109× |
| Clusters 4096×128 | 151 | fit_query | 1.0465× | 1.0690× | 1.1010× | 1.1206× |
| Clusters 4096×128 | 251 | kneighbors | 1.0158× | 1.0277× | 1.0297× | 1.0385× |
| Clusters 4096×128 | 251 | fit_query | 1.1505× | 1.1306× | 1.1380× | 1.0734× |
| Clusters 4096×128 | 252 | kneighbors | 1.0048× | 1.0174× | 1.0288× | 1.0338× |
| Clusters 4096×128 | 252 | fit_query | 1.0537× | 1.0613× | 1.0821× | 1.0787× |

## Every cell against Current forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 4096×128 | 151 | kneighbors | **0.9589×** | **0.9899×** | **0.9936×** | 1.0010× |
| Gaussian 4096×128 | 151 | fit_query | **0.9977×** | 1.0304× | **0.9765×** | 1.0006× |
| Gaussian 4096×128 | 251 | kneighbors | 1.0310× | 1.0034× | 1.0203× | 1.0087× |
| Gaussian 4096×128 | 251 | fit_query | **0.9562×** | **0.9343×** | 1.0285× | 1.0190× |
| Gaussian 4096×128 | 252 | kneighbors | 1.0059× | 1.0256× | 1.0309× | 1.0294× |
| Gaussian 4096×128 | 252 | fit_query | 1.0152× | **0.9789×** | **0.9695×** | **0.9723×** |
| Gaussian 4096×128 | 351 | kneighbors | **0.9512×** | **0.9632×** | **0.9388×** | **0.9670×** |
| Gaussian 4096×128 | 351 | fit_query | **0.9583×** | **0.9671×** | **0.9567×** | **0.9848×** |
| Gaussian 4096×128 | 451 | kneighbors | **0.9548×** | **0.9455×** | 1.0131× | **0.9987×** |
| Gaussian 4096×128 | 451 | fit_query | 1.0002× | **0.9599×** | 1.0061× | 1.0051× |
| Gaussian 4096×128 | 552 | kneighbors | 1.0619× | 1.0121× | 1.0281× | 1.0263× |
| Gaussian 4096×128 | 552 | fit_query | **0.9799×** | 1.0176× | 1.0120× | **0.9859×** |
| Clusters 4096×128 | 151 | kneighbors | **0.9831×** | **0.9866×** | **0.9672×** | **0.9983×** |
| Clusters 4096×128 | 151 | fit_query | **0.9819×** | 1.0041× | 1.0331× | 1.0379× |
| Clusters 4096×128 | 251 | kneighbors | **0.9920×** | 1.0062× | 1.0056× | 1.0149× |
| Clusters 4096×128 | 251 | fit_query | 1.0082× | 1.0019× | **0.9973×** | **0.9910×** |
| Clusters 4096×128 | 252 | kneighbors | **0.9602×** | **0.9507×** | **0.9832×** | **0.9690×** |
| Clusters 4096×128 | 252 | fit_query | **0.9775×** | **0.9848×** | 1.0039× | **0.9960×** |

## Every cell against Historical forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 4096×128 | 151 | kneighbors | **0.9454×** | **0.9530×** | **0.9797×** | **0.9577×** |
| Gaussian 4096×128 | 151 | fit_query | 1.0438× | 1.0958× | 1.0216× | 1.0717× |
| Gaussian 4096×128 | 251 | kneighbors | 1.0505× | 1.0316× | 1.0396× | **0.9949×** |
| Gaussian 4096×128 | 251 | fit_query | **0.9490×** | 1.0461× | 1.0207× | 1.1333× |
| Gaussian 4096×128 | 252 | kneighbors | 1.0252× | 1.0145× | 1.0506× | 1.0583× |
| Gaussian 4096×128 | 252 | fit_query | 1.0801× | 1.0728× | 1.0315× | 1.0750× |
| Gaussian 4096×128 | 351 | kneighbors | **0.9829×** | **0.9768×** | **0.9700×** | **0.9780×** |
| Gaussian 4096×128 | 351 | fit_query | 1.0108× | 1.0546× | 1.0092× | 1.0986× |
| Gaussian 4096×128 | 451 | kneighbors | **0.9686×** | **0.9825×** | 1.0278× | 1.0343× |
| Gaussian 4096×128 | 451 | fit_query | **0.9878×** | 1.0197× | **0.9936×** | **0.9836×** |
| Gaussian 4096×128 | 552 | kneighbors | **0.9754×** | **0.9706×** | **0.9444×** | **0.9476×** |
| Gaussian 4096×128 | 552 | fit_query | 1.0177× | 1.1023× | 1.0510× | 1.0268× |
| Clusters 4096×128 | 151 | kneighbors | **0.9779×** | 1.0069× | **0.9620×** | **0.9658×** |
| Clusters 4096×128 | 151 | fit_query | 1.0299× | 1.0490× | 1.0836× | 1.0910× |
| Clusters 4096×128 | 251 | kneighbors | **0.9853×** | **0.9931×** | **0.9988×** | **0.9983×** |
| Clusters 4096×128 | 251 | fit_query | 1.0734× | 1.0664× | 1.0618× | 1.0456× |
| Clusters 4096×128 | 252 | kneighbors | 1.0002× | **0.9974×** | 1.0241× | 1.0283× |
| Clusters 4096×128 | 252 | fit_query | 1.0464× | 1.0613× | 1.0746× | 1.0908× |

## Accuracy and source identity

All cold, warmup, probe and retained strict audits remain in the raw evidence. Every worker passes; no accuracy failure filters a timing denominator. Each repeat fixture and oracle record exactly matches the corresponding full-control fixture. Source, runtime and harness archive bytes, measured worker identities, and current source/native/harness files have been independently verified.

| Worker | Cold passes/audits | Warmup passes/audits | Probe passes/audits | Retained passes/audits |
| --- | --- | --- | --- | --- |
| Current auto | 9/9 | 32316/32316 | 18/18 | 558/558 |
| Current auto A/A | 9/9 | 31793/31793 | 18/18 | 558/558 |
| Current forced GPU | 9/9 | 32448/32448 | 18/18 | 558/558 |
| Immediate prior auto | 9/9 | 31709/31709 | 18/18 | 558/558 |
| Historical forced GPU | 9/9 | 32022/32022 | 18/18 | 558/558 |

All raw paired ratios, actual routing, first/middle/last summaries, position and predecessor diagnostics remain in the independent audit. They diagnose timing sensitivity without claiming an identified clock mechanism.

## Evidence

[Every raw observation](../benchmarks/results/knn-startup-final2-2026-10-30-preservation-repeat.json) — SHA-256 `20ce3b67688ca2e402710ecd541a95ebe6070c1d87977ac15673d79136f8345e`.

[Independent audit and complete crossmatching](../benchmarks/results/knn-startup-final2-2026-10-30-preservation-repeat-audit.json).

[Full balanced control](../benchmarks/results/knn-startup-final2-2026-10-30-control.json).
