# Predeclared Gaussian and clustered 128D preservation repeat

All **18 operation cells**, both automatic replicas and all five workers are retained. Accuracy, runtime identity and numerical-oracle validation pass. The panel-specific crosschecks below retain both the full result and the repeat, including smaller or inconsistent slowdowns. Numerical acceptance is separate from the declared Digits performance screen.

This is the `preservation-repeat` portion of the [saved predeclaration](../benchmarks/results/knn-digits-final-predeclaration-2026-10-30.json) (SHA-256 `68a31e8f20c4dfe5161fd0df852ab6a208c15a73dd3099a5a5374670642b2b63`). Both nine-fixture inventories were retained from the preceding final validation and declared before candidate GPU controls and full panels. Baseline profiling or native ablations may already have occurred. The exact requested inventories are verified, with no seed or operation removed.

Each operation uses 31 serial interleaved rounds across current auto, identical current auto A/A, current forced GPU, immediate prior auto (`knn-startup-final2-2026-10-30` plus its exact archived native runtime), and historical forced GPU (`knn-small-final-2026-10-30`). This panel uses the legacy forward/reverse ordering and 1 second of continuous warmup. The full 48-fixture control uses balanced Williams ordering and 0.25 seconds of warmup. Absolute milliseconds across these runs are not directly comparable; ratios compare implementations within their own run. Neither run replaces the other, and no faster window or run is selected.

M/P means ratio of medians / median paired ratio. Ratios are comparator/current; below 1× means the automatic replica is slower and is bold. A >5% runtime slowdown means ratio <1/1.05 for BOTH statistics and BOTH independent automatic replicas. This recurrence screen is descriptive, not a confidence interval or significance test.

## Full-control exceptions and repeat crosscheck

| Comparator | Flags in full 222 cells | Covered here | Recurring | Not recurring | New flags |
| --- | --- | --- | --- | --- | --- |
| Immediate prior auto | 18 | 0 | 0 | 0 | 0 |
| Current forced GPU | 1 | 1 | 0 | 1 | 0 |
| Historical forced GPU | 0 | 0 | 0 | 0 | 1 |

| Comparator | Classification | Family | Seed | Operation | Full current M/P | Full A/A M/P | Repeat current M/P | Repeat A/A M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Current forced GPU | not recurring | Gaussian 4096×128 | 552 | kneighbors | **0.9266×** / **0.9167×** | **0.9393×** / **0.9374×** | **0.9724×** / **0.9703×** | **0.9770×** / **0.9977×** |
| Historical forced GPU | new in repeat | Gaussian 4096×128 | 552 | kneighbors | 1.0059× / **0.9948×** | 1.0197× / 1.0257× | **0.9405×** / **0.9470×** | **0.9449×** / **0.9427×** |

## Every slower observation in either statistic

| Comparator | Replica | Family | Seed | Operation | M/P |
| --- | --- | --- | --- | --- | --- |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 151 | kneighbors | 1.0076× / **0.9816×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 151 | kneighbors | 1.0025× / **0.9819×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 151 | kneighbors | **0.9475×** / **0.9439×** |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 151 | kneighbors | **0.9790×** / **0.9730×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 151 | fit_query | **0.9911×** / **0.9722×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 151 | fit_query | **0.9933×** / 1.0012× |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 251 | kneighbors | **0.9858×** / **0.9901×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 251 | kneighbors | 1.0189× / **0.9832×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 251 | fit_query | 1.0140× / **0.9910×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 251 | fit_query | 1.0081× / **0.9895×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 252 | fit_query | 1.0130× / **0.9816×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 252 | fit_query | 1.0471× / **0.9960×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 351 | kneighbors | **0.9908×** / **0.9888×** |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 351 | kneighbors | **0.9926×** / 1.0237× |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 351 | fit_query | **0.9789×** / **0.9589×** |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 351 | fit_query | 1.0076× / **0.9889×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 351 | fit_query | **0.9982×** / **0.9482×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 351 | fit_query | 1.0276× / **0.9833×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 451 | kneighbors | 1.0137× / **0.9167×** |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 451 | kneighbors | 1.0158× / **0.9548×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 451 | kneighbors | 1.0360× / **0.9787×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 451 | kneighbors | 1.0843× / **0.9654×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 451 | fit_query | **0.9379×** / **0.9760×** |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 451 | fit_query | **0.9855×** / 1.0247× |
| Current forced GPU | Current auto | Gaussian 4096×128 | 451 | fit_query | **0.9431×** / **0.9925×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 451 | fit_query | **0.9909×** / **0.9930×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 451 | fit_query | **0.9980×** / **0.9616×** |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 451 | fit_query | 1.0486× / **0.9780×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 552 | kneighbors | **0.9967×** / **0.9894×** |
| Current forced GPU | Current auto | Gaussian 4096×128 | 552 | kneighbors | **0.9724×** / **0.9703×** |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 552 | kneighbors | **0.9770×** / **0.9977×** |
| Historical forced GPU | Current auto | Gaussian 4096×128 | 552 | kneighbors | **0.9405×** / **0.9470×** |
| Historical forced GPU | Current auto A/A | Gaussian 4096×128 | 552 | kneighbors | **0.9449×** / **0.9427×** |
| Immediate prior auto | Current auto | Gaussian 4096×128 | 552 | fit_query | 1.0076× / **0.9953×** |
| Immediate prior auto | Current auto A/A | Gaussian 4096×128 | 552 | fit_query | **0.9769×** / 1.0192× |
| Current forced GPU | Current auto A/A | Gaussian 4096×128 | 552 | fit_query | **0.9820×** / 1.0097× |
| Immediate prior auto | Current auto A/A | Clusters 4096×128 | 151 | kneighbors | **0.9997×** / 1.0138× |
| Current forced GPU | Current auto A/A | Clusters 4096×128 | 151 | kneighbors | **0.9977×** / 1.0193× |
| Historical forced GPU | Current auto A/A | Clusters 4096×128 | 151 | kneighbors | **0.9829×** / **0.9839×** |
| Immediate prior auto | Current auto | Clusters 4096×128 | 151 | fit_query | **0.9957×** / 1.0066× |
| Current forced GPU | Current auto | Clusters 4096×128 | 251 | kneighbors | **0.9908×** / 1.0177× |
| Current forced GPU | Current auto | Clusters 4096×128 | 251 | fit_query | **0.9988×** / **0.9618×** |
| Current forced GPU | Current auto A/A | Clusters 4096×128 | 251 | fit_query | 1.0236× / **0.9674×** |
| Current forced GPU | Current auto | Clusters 4096×128 | 252 | kneighbors | **0.9917×** / **0.9749×** |
| Historical forced GPU | Current auto | Clusters 4096×128 | 252 | kneighbors | **0.9928×** / 1.0009× |
| Immediate prior auto | Current auto | Clusters 4096×128 | 252 | fit_query | **0.9648×** / **0.9955×** |
| Immediate prior auto | Current auto A/A | Clusters 4096×128 | 252 | fit_query | **0.9812×** / 1.0088× |
| Current forced GPU | Current auto | Clusters 4096×128 | 252 | fit_query | **0.9868×** / **0.9998×** |

## Every repeat timing cell

All times are median milliseconds from this repeat only.

| Family | Seed | Operation | Current auto ms | Current auto A/A ms | Current forced GPU ms | Immediate prior auto ms | Historical forced GPU ms | A/A over auto M/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 4096×128 | 151 | kneighbors | 0.412792 | 0.399500 | 0.413833 | 0.415917 | 0.391125 | **0.9678×** / **0.9740×** |
| Gaussian 4096×128 | 151 | fit_query | 0.603833 | 0.590709 | 0.599791 | 0.598459 | 0.625625 | **0.9783×** / **0.9544×** |
| Gaussian 4096×128 | 251 | kneighbors | 0.379084 | 0.367500 | 0.387125 | 0.373708 | 0.386250 | **0.9694×** / 1.0005× |
| Gaussian 4096×128 | 251 | fit_query | 0.649333 | 0.614542 | 0.654583 | 0.658417 | 0.682750 | **0.9464×** / **0.9541×** |
| Gaussian 4096×128 | 252 | kneighbors | 0.400750 | 0.382750 | 0.413125 | 0.401584 | 0.401625 | **0.9551×** / **0.9923×** |
| Gaussian 4096×128 | 252 | fit_query | 0.622208 | 0.601958 | 0.630292 | 0.684042 | 0.692500 | **0.9675×** / **0.9795×** |
| Gaussian 4096×128 | 351 | kneighbors | 0.392833 | 0.392125 | 0.400125 | 0.404250 | 0.389208 | **0.9982×** / **0.9880×** |
| Gaussian 4096×128 | 351 | fit_query | 0.612834 | 0.595333 | 0.611750 | 0.599875 | 0.648125 | **0.9714×** / **0.9712×** |
| Gaussian 4096×128 | 451 | kneighbors | 0.460750 | 0.459833 | 0.477333 | 0.467084 | 0.499583 | **0.9980×** / **0.9782×** |
| Gaussian 4096×128 | 451 | fit_query | 0.656541 | 0.624875 | 0.619208 | 0.615792 | 0.655250 | **0.9518×** / **0.9905×** |
| Gaussian 4096×128 | 552 | kneighbors | 0.392583 | 0.390750 | 0.381750 | 0.391292 | 0.369209 | **0.9953×** / 1.0016× |
| Gaussian 4096×128 | 552 | fit_query | 0.592750 | 0.611375 | 0.600375 | 0.597250 | 0.620875 | 1.0314× / 1.0071× |
| Clusters 4096×128 | 151 | kneighbors | 0.382625 | 0.394042 | 0.393125 | 0.393917 | 0.387292 | 1.0298× / 1.0159× |
| Clusters 4096×128 | 151 | fit_query | 0.533375 | 0.510500 | 0.534458 | 0.531084 | 0.564084 | **0.9571×** / **0.9532×** |
| Clusters 4096×128 | 251 | kneighbors | 0.396584 | 0.379458 | 0.392917 | 0.403292 | 0.398833 | **0.9568×** / **0.9686×** |
| Clusters 4096×128 | 251 | fit_query | 0.572667 | 0.558833 | 0.572000 | 0.583291 | 0.608333 | **0.9758×** / **0.9783×** |
| Clusters 4096×128 | 252 | kneighbors | 0.395708 | 0.390792 | 0.392416 | 0.403292 | 0.392875 | **0.9876×** / **0.9720×** |
| Clusters 4096×128 | 252 | fit_query | 0.544417 | 0.535292 | 0.537250 | 0.525250 | 0.560292 | **0.9832×** / **0.9867×** |

## Every cell against Immediate prior auto

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 4096×128 | 151 | kneighbors | 1.0076× | **0.9816×** | 1.0411× | 1.0102× |
| Gaussian 4096×128 | 151 | fit_query | **0.9911×** | **0.9722×** | 1.0131× | 1.0233× |
| Gaussian 4096×128 | 251 | kneighbors | **0.9858×** | **0.9901×** | 1.0169× | 1.0163× |
| Gaussian 4096×128 | 251 | fit_query | 1.0140× | **0.9910×** | 1.0714× | 1.0620× |
| Gaussian 4096×128 | 252 | kneighbors | 1.0021× | 1.0331× | 1.0492× | 1.0439× |
| Gaussian 4096×128 | 252 | fit_query | 1.0994× | 1.0080× | 1.1364× | 1.0810× |
| Gaussian 4096×128 | 351 | kneighbors | 1.0291× | 1.0133× | 1.0309× | 1.0370× |
| Gaussian 4096×128 | 351 | fit_query | **0.9789×** | **0.9589×** | 1.0076× | **0.9889×** |
| Gaussian 4096×128 | 451 | kneighbors | 1.0137× | **0.9167×** | 1.0158× | **0.9548×** |
| Gaussian 4096×128 | 451 | fit_query | **0.9379×** | **0.9760×** | **0.9855×** | 1.0247× |
| Gaussian 4096×128 | 552 | kneighbors | **0.9967×** | **0.9894×** | 1.0014× | 1.0000× |
| Gaussian 4096×128 | 552 | fit_query | 1.0076× | **0.9953×** | **0.9769×** | 1.0192× |
| Clusters 4096×128 | 151 | kneighbors | 1.0295× | 1.0362× | **0.9997×** | 1.0138× |
| Clusters 4096×128 | 151 | fit_query | **0.9957×** | 1.0066× | 1.0403× | 1.0564× |
| Clusters 4096×128 | 251 | kneighbors | 1.0169× | 1.0169× | 1.0628× | 1.0486× |
| Clusters 4096×128 | 251 | fit_query | 1.0186× | 1.0275× | 1.0438× | 1.0236× |
| Clusters 4096×128 | 252 | kneighbors | 1.0192× | 1.0035× | 1.0320× | 1.0404× |
| Clusters 4096×128 | 252 | fit_query | **0.9648×** | **0.9955×** | **0.9812×** | 1.0088× |

## Every cell against Current forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 4096×128 | 151 | kneighbors | 1.0025× | **0.9819×** | 1.0359× | 1.0045× |
| Gaussian 4096×128 | 151 | fit_query | **0.9933×** | 1.0012× | 1.0154× | 1.0133× |
| Gaussian 4096×128 | 251 | kneighbors | 1.0212× | 1.0090× | 1.0534× | 1.0262× |
| Gaussian 4096×128 | 251 | fit_query | 1.0081× | **0.9895×** | 1.0652× | 1.0346× |
| Gaussian 4096×128 | 252 | kneighbors | 1.0309× | 1.0309× | 1.0794× | 1.0393× |
| Gaussian 4096×128 | 252 | fit_query | 1.0130× | **0.9816×** | 1.0471× | **0.9960×** |
| Gaussian 4096×128 | 351 | kneighbors | 1.0186× | 1.0047× | 1.0204× | 1.0098× |
| Gaussian 4096×128 | 351 | fit_query | **0.9982×** | **0.9482×** | 1.0276× | **0.9833×** |
| Gaussian 4096×128 | 451 | kneighbors | 1.0360× | **0.9787×** | 1.0381× | 1.0289× |
| Gaussian 4096×128 | 451 | fit_query | **0.9431×** | **0.9925×** | **0.9909×** | **0.9930×** |
| Gaussian 4096×128 | 552 | kneighbors | **0.9724×** | **0.9703×** | **0.9770×** | **0.9977×** |
| Gaussian 4096×128 | 552 | fit_query | 1.0129× | 1.0184× | **0.9820×** | 1.0097× |
| Clusters 4096×128 | 151 | kneighbors | 1.0274× | 1.0384× | **0.9977×** | 1.0193× |
| Clusters 4096×128 | 151 | fit_query | 1.0020× | 1.0283× | 1.0469× | 1.0837× |
| Clusters 4096×128 | 251 | kneighbors | **0.9908×** | 1.0177× | 1.0355× | 1.0482× |
| Clusters 4096×128 | 251 | fit_query | **0.9988×** | **0.9618×** | 1.0236× | **0.9674×** |
| Clusters 4096×128 | 252 | kneighbors | **0.9917×** | **0.9749×** | 1.0042× | 1.0012× |
| Clusters 4096×128 | 252 | fit_query | **0.9868×** | **0.9998×** | 1.0037× | 1.0198× |

## Every cell against Historical forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 4096×128 | 151 | kneighbors | **0.9475×** | **0.9439×** | **0.9790×** | **0.9730×** |
| Gaussian 4096×128 | 151 | fit_query | 1.0361× | 1.0720× | 1.0591× | 1.0717× |
| Gaussian 4096×128 | 251 | kneighbors | 1.0189× | **0.9832×** | 1.0510× | 1.0412× |
| Gaussian 4096×128 | 251 | fit_query | 1.0515× | 1.0457× | 1.1110× | 1.0960× |
| Gaussian 4096×128 | 252 | kneighbors | 1.0022× | 1.0098× | 1.0493× | 1.0432× |
| Gaussian 4096×128 | 252 | fit_query | 1.1130× | 1.0993× | 1.1504× | 1.1011× |
| Gaussian 4096×128 | 351 | kneighbors | **0.9908×** | **0.9888×** | **0.9926×** | 1.0237× |
| Gaussian 4096×128 | 351 | fit_query | 1.0576× | 1.0379× | 1.0887× | 1.0764× |
| Gaussian 4096×128 | 451 | kneighbors | 1.0843× | **0.9654×** | 1.0864× | 1.0137× |
| Gaussian 4096×128 | 451 | fit_query | **0.9980×** | **0.9616×** | 1.0486× | **0.9780×** |
| Gaussian 4096×128 | 552 | kneighbors | **0.9405×** | **0.9470×** | **0.9449×** | **0.9427×** |
| Gaussian 4096×128 | 552 | fit_query | 1.0474× | 1.0556× | 1.0155× | 1.0163× |
| Clusters 4096×128 | 151 | kneighbors | 1.0122× | 1.0122× | **0.9829×** | **0.9839×** |
| Clusters 4096×128 | 151 | fit_query | 1.0576× | 1.0514× | 1.1050× | 1.0867× |
| Clusters 4096×128 | 251 | kneighbors | 1.0057× | 1.0059× | 1.0511× | 1.0341× |
| Clusters 4096×128 | 251 | fit_query | 1.0623× | 1.0647× | 1.0886× | 1.0700× |
| Clusters 4096×128 | 252 | kneighbors | **0.9928×** | 1.0009× | 1.0053× | 1.0083× |
| Clusters 4096×128 | 252 | fit_query | 1.0292× | 1.0541× | 1.0467× | 1.0938× |

## Accuracy and source identity

All cold, warmup, probe and retained strict audits remain in the raw evidence. Every worker passes; no accuracy failure filters a timing denominator. Each repeat fixture and oracle record exactly matches the corresponding full-control fixture. Source, runtime and harness archive bytes, measured worker identities, and current source/native/harness files have been independently verified.

| Worker | Cold passes/audits | Warmup passes/audits | Probe passes/audits | Retained passes/audits |
| --- | --- | --- | --- | --- |
| Current auto | 9/9 | 31887/31887 | 18/18 | 558/558 |
| Current auto A/A | 9/9 | 31695/31695 | 18/18 | 558/558 |
| Current forced GPU | 9/9 | 31049/31049 | 18/18 | 558/558 |
| Immediate prior auto | 9/9 | 31897/31897 | 18/18 | 558/558 |
| Historical forced GPU | 9/9 | 31408/31408 | 18/18 | 558/558 |

All raw paired ratios, actual routing, first/middle/last summaries, position and predecessor diagnostics remain in the independent audit. They diagnose timing sensitivity without claiming an identified clock mechanism.

## Evidence

[Every raw observation](../benchmarks/results/knn-digits-final-2026-10-30-preservation-repeat.json) — SHA-256 `bd2627a521e5eb1c5abba9fa6251dc0032bc2b5e4f50fd4550f2267becb91dda`.

[Independent audit and complete crossmatching](../benchmarks/results/knn-digits-final-2026-10-30-preservation-repeat-audit.json).

[Full balanced control](../benchmarks/results/knn-digits-final-2026-10-30-control.json).
