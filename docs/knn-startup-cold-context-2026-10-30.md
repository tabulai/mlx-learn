# Fresh-process KNN first-use benchmark

Completed `2026-10-30T23:41:57.702029+00:00`. This report retains **450 separate process launches** over 18 fixtures, 5 configurations and 5 launches per fixture/configuration. Accuracy/integrity acceptance: **True**.

Each retained launch starts a new interpreter and performs exactly one first public NearestNeighbors fit/query. There are no warmups, metadata preflight requests, harness-driven native preimports or external pre-fit MLX synchronization. NumPy import and fixture X/Q loading are reported separately from estimator import/setup and API calls. Necessary transitive library imports remain charged. The first stdout result marker follows returned NumPy neighbors, before audits and runtime metadata. Parent process-start-to-result includes interpreter startup, fixture input loading, estimator imports, fit/query and result-marker IPC. Process elapsed also includes post-result audits/metadata/exit. This is process-cold, not a claim of cold OS/filesystem/Metal disk caches. All launches are serial.

Fit/query reductions may reflect moving native initialization into normal estimator imports. Estimator setup and estimator/process-to-result measures retain that cost. Parent result-marker latency includes interpreter/input loading/IPC; worker estimator-to-result excludes input preparation. No launch or failed audit is removed, and no OS/bytecode/Metal-disk-cache coldness is claimed.

All times are milliseconds. Speedup is comparator time / current automatic time; a value below 1× means current is slower. Bold ratios mark every slower or missing median. No failed numerical audit removes a timing from the comparison. These are descriptive process-cold measurements, not a universal first-use latency guarantee or a machine-cold test.

## Coverage by separately measured phase

| Phase | Comparator | Current medians faster | Minimum speedup | Maximum speedup |
| --- | --- | --- | --- | --- |
| First query | Default-thread brute | 18/18 | 22.907× | 169.002× |
| First query | Single-thread brute | 18/18 | 10.511× | 73.495× |
| First query | sklearn auto | 18/18 | 3.260× | 101.961× |
| First query | Preceding automatic | 14/18 | **0.956×** | 1.232× |
| First fit + query | Default-thread brute | 18/18 | 20.152× | 82.728× |
| First fit + query | Single-thread brute | 18/18 | 9.580× | 37.491× |
| First fit + query | sklearn auto | 18/18 | 3.952× | 45.239× |
| First fit + query | Preceding automatic | 17/18 | **0.969×** | 1.365× |
| Estimator import and setup | Default-thread brute | 18/18 | 1.175× | 1.208× |
| Estimator import and setup | Single-thread brute | 18/18 | 1.196× | 1.226× |
| Estimator import and setup | sklearn auto | 18/18 | 1.181× | 1.208× |
| Estimator import and setup | Preceding automatic | 18/18 | 1.196× | 1.226× |
| Estimator import through first result | Default-thread brute | 18/18 | 1.196× | 1.228× |
| Estimator import through first result | Single-thread brute | 18/18 | 1.205× | 1.235× |
| Estimator import through first result | sklearn auto | 18/18 | 1.190× | 1.229× |
| Estimator import through first result | Preceding automatic | 18/18 | 1.196× | 1.226× |
| Process start through first result | Default-thread brute | 18/18 | 1.175× | 1.210× |
| Process start through first result | Single-thread brute | 18/18 | 1.185× | 1.221× |
| Process start through first result | sklearn auto | 18/18 | 1.176× | 1.212× |
| Process start through first result | Preceding automatic | 18/18 | 1.181× | 1.213× |

## First query

First public query after the newly constructed estimator's first fit.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.089 | 0.080–0.092 | 13.363 / 150.852× | 6.000 / 67.733× | 0.289 / 3.260× | 0.101 / 1.145× |
| Gaussian 32D, single query | 49 | 0.134 | 0.126–0.149 | 13.553 / 101.331× | 5.513 / 41.220× | 13.637 / 101.961× | 0.136 / 1.018× |
| Gaussian 8D, single query | 42 | 0.088 | 0.080–0.089 | 13.245 / 150.372× | 5.858 / 66.509× | 0.297 / 3.376× | 0.095 / 1.082× |
| Gaussian 32D, single query | 50 | 0.139 | 0.123–0.146 | 13.266 / 95.183× | 5.913 / 42.425× | 13.731 / 98.518× | 0.136 / **0.975×** |
| Gaussian 8D, single query | 43 | 0.082 | 0.080–0.091 | 12.858 / 156.724× | 5.679 / 69.219× | 0.301 / 3.668× | 0.098 / 1.196× |
| Gaussian 32D, single query | 51 | 0.148 | 0.146–0.168 | 13.470 / 91.039× | 5.527 / 37.356× | 13.750 / 92.933× | 0.149 / 1.008× |
| Digits (raw) | 151 | 0.614 | 0.586–0.661 | 14.055 / 22.907× | 6.942 / 11.314× | 14.158 / 23.074× | 0.587 / **0.956×** |
| Digits (raw) | 251 | 0.601 | 0.590–0.632 | 14.376 / 23.923× | 6.316 / 10.511× | 14.155 / 23.555× | 0.601 / **1.000×** |
| Digits (raw) | 252 | 0.594 | 0.564–0.615 | 13.893 / 23.396× | 6.902 / 11.623× | 14.130 / 23.796× | 0.585 / **0.985×** |
| Breast Cancer (raw) | 151 | 0.224 | 0.220–0.240 | 13.209 / 58.979× | 5.440 / 24.290× | 13.587 / 60.668× | 0.276 / 1.232× |
| Breast Cancer (raw) | 251 | 0.229 | 0.219–0.240 | 13.702 / 59.877× | 5.788 / 25.293× | 13.284 / 58.052× | 0.267 / 1.166× |
| Breast Cancer (raw) | 252 | 0.234 | 0.233–0.238 | 13.531 / 57.814× | 5.680 / 24.270× | 13.337 / 56.986× | 0.275 / 1.175× |
| Iris (raw) | 151 | 0.081 | 0.070–0.114 | 12.815 / 158.053× | 5.628 / 69.414× | 0.289 / 3.565× | 0.082 / 1.009× |
| Iris (raw) | 251 | 0.076 | 0.073–0.080 | 12.922 / 169.002× | 5.610 / 73.374× | 0.298 / 3.893× | 0.081 / 1.056× |
| Iris (raw) | 252 | 0.077 | 0.069–0.083 | 12.906 / 168.255× | 5.638 / 73.495× | 0.288 / 3.754× | 0.083 / 1.077× |
| Wine (raw) | 151 | 0.087 | 0.083–0.095 | 13.363 / 152.934× | 5.867 / 67.151× | 0.307 / 3.510× | 0.095 / 1.088× |
| Wine (raw) | 251 | 0.086 | 0.076–0.093 | 13.080 / 152.606× | 5.959 / 69.531× | 0.292 / 3.409× | 0.091 / 1.066× |
| Wine (raw) | 252 | 0.089 | 0.086–0.090 | 13.122 / 147.778× | 5.660 / 63.740× | 0.291 / 3.280× | 0.093 / 1.046× |

## First fit + query

Constructor, first fit and first query. Estimator imports/setup are separate.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.275 | 0.252–0.284 | 13.666 / 49.634× | 6.331 / 22.995× | 1.994 / 7.244× | 0.324 / 1.176× |
| Gaussian 32D, single query | 49 | 0.351 | 0.331–0.369 | 13.889 / 39.529× | 5.853 / 16.659× | 13.975 / 39.771× | 0.362 / 1.029× |
| Gaussian 8D, single query | 42 | 0.260 | 0.259–0.266 | 13.547 / 52.130× | 6.189 / 23.817× | 2.067 / 7.953× | 0.310 / 1.195× |
| Gaussian 32D, single query | 50 | 0.356 | 0.325–0.404 | 13.589 / 38.215× | 6.268 / 17.627× | 14.052 / 39.518× | 0.379 / 1.066× |
| Gaussian 8D, single query | 43 | 0.259 | 0.244–0.277 | 13.152 / 50.870× | 6.003 / 23.219× | 2.052 / 7.936× | 0.314 / 1.216× |
| Gaussian 32D, single query | 51 | 0.375 | 0.366–0.408 | 13.824 / 36.892× | 5.885 / 15.704× | 14.117 / 37.675× | 0.363 / **0.969×** |
| Digits (raw) | 151 | 0.714 | 0.685–0.756 | 14.388 / 20.152× | 7.291 / 10.212× | 14.476 / 20.276× | 0.727 / 1.018× |
| Digits (raw) | 251 | 0.695 | 0.683–0.726 | 14.695 / 21.159× | 6.653 / 9.580× | 14.468 / 20.832× | 0.740 / 1.065× |
| Digits (raw) | 252 | 0.689 | 0.652–0.712 | 14.199 / 20.606× | 7.228 / 10.489× | 14.450 / 20.970× | 0.729 / 1.058× |
| Breast Cancer (raw) | 151 | 0.307 | 0.306–0.328 | 13.523 / 44.043× | 5.803 / 18.899× | 13.890 / 45.239× | 0.400 / 1.301× |
| Breast Cancer (raw) | 251 | 0.322 | 0.303–0.327 | 14.013 / 43.502× | 6.119 / 18.995× | 13.592 / 42.195× | 0.393 / 1.221× |
| Breast Cancer (raw) | 252 | 0.318 | 0.316–0.327 | 13.825 / 43.469× | 6.011 / 18.899× | 13.639 / 42.883× | 0.400 / 1.259× |
| Iris (raw) | 151 | 0.166 | 0.148–0.196 | 13.127 / 78.863× | 5.950 / 35.745× | 0.679 / 4.077× | 0.199 / 1.197× |
| Iris (raw) | 251 | 0.160 | 0.153–0.176 | 13.243 / 82.728× | 5.950 / 37.170× | 0.692 / 4.325× | 0.206 / 1.284× |
| Iris (raw) | 252 | 0.160 | 0.153–0.182 | 13.211 / 82.722× | 5.975 / 37.415× | 0.683 / 4.274× | 0.206 / 1.293× |
| Wine (raw) | 151 | 0.170 | 0.158–0.201 | 13.671 / 80.478× | 6.201 / 36.504× | 0.704 / 4.143× | 0.232 / 1.365× |
| Wine (raw) | 251 | 0.168 | 0.152–0.192 | 13.391 / 79.928× | 6.281 / 37.491× | 0.696 / 4.152× | 0.215 / 1.286× |
| Wine (raw) | 252 | 0.174 | 0.164–0.176 | 13.419 / 77.159× | 6.039 / 34.725× | 0.687 / 3.952× | 0.218 / 1.253× |

## Estimator import and setup

Normal estimator dependencies and configuration setup, including native initialization loaded by those imports.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 625.797 | 621.169–641.293 | 744.750 / 1.190× | 753.445 / 1.204× | 753.588 / 1.204× | 754.522 / 1.206× |
| Gaussian 32D, single query | 49 | 630.153 | 620.506–640.331 | 760.056 / 1.206× | 764.693 / 1.214× | 752.998 / 1.195× | 768.081 / 1.219× |
| Gaussian 8D, single query | 42 | 635.333 | 628.906–639.082 | 759.011 / 1.195× | 773.258 / 1.217× | 763.429 / 1.202× | 760.115 / 1.196× |
| Gaussian 32D, single query | 50 | 633.888 | 627.827–647.330 | 761.521 / 1.201× | 766.829 / 1.210× | 759.042 / 1.197× | 777.413 / 1.226× |
| Gaussian 8D, single query | 43 | 637.364 | 629.149–654.509 | 758.073 / 1.189× | 776.736 / 1.219× | 768.083 / 1.205× | 771.630 / 1.211× |
| Gaussian 32D, single query | 51 | 646.197 | 632.804–662.640 | 759.195 / 1.175× | 772.964 / 1.196× | 763.426 / 1.181× | 784.616 / 1.214× |
| Digits (raw) | 151 | 640.453 | 633.230–651.421 | 768.194 / 1.199× | 775.082 / 1.210× | 762.171 / 1.190× | 780.226 / 1.218× |
| Digits (raw) | 251 | 633.839 | 630.386–641.047 | 759.136 / 1.198× | 768.598 / 1.213× | 765.460 / 1.208× | 766.443 / 1.209× |
| Digits (raw) | 252 | 634.034 | 629.160–639.442 | 762.569 / 1.203× | 777.465 / 1.226× | 764.558 / 1.206× | 774.285 / 1.221× |
| Breast Cancer (raw) | 151 | 634.306 | 630.468–653.782 | 765.800 / 1.207× | 775.199 / 1.222× | 762.026 / 1.201× | 774.094 / 1.220× |
| Breast Cancer (raw) | 251 | 634.721 | 632.826–643.657 | 760.746 / 1.199× | 773.958 / 1.219× | 758.909 / 1.196× | 774.899 / 1.221× |
| Breast Cancer (raw) | 252 | 635.649 | 629.655–645.183 | 761.738 / 1.198× | 774.467 / 1.218× | 760.826 / 1.197× | 766.649 / 1.206× |
| Iris (raw) | 151 | 635.154 | 632.291–642.751 | 761.501 / 1.199× | 775.316 / 1.221× | 766.047 / 1.206× | 772.949 / 1.217× |
| Iris (raw) | 251 | 638.381 | 632.902–639.476 | 761.366 / 1.193× | 779.139 / 1.220× | 763.818 / 1.196× | 780.224 / 1.222× |
| Iris (raw) | 252 | 640.043 | 634.176–651.856 | 762.472 / 1.191× | 768.329 / 1.200× | 761.288 / 1.189× | 768.321 / 1.200× |
| Wine (raw) | 151 | 635.888 | 631.984–644.818 | 758.828 / 1.193× | 777.514 / 1.223× | 762.231 / 1.199× | 772.596 / 1.215× |
| Wine (raw) | 251 | 635.441 | 630.049–640.135 | 764.264 / 1.203× | 775.595 / 1.221× | 763.792 / 1.202× | 767.374 / 1.208× |
| Wine (raw) | 252 | 638.640 | 633.543–640.482 | 771.235 / 1.208× | 775.491 / 1.214× | 768.316 / 1.203× | 775.908 / 1.215× |

## Estimator import through first result

Estimator import/setup plus first fit/query. Moving native initialization into imports does not remove it from this measure.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 626.072 | 621.421–641.573 | 758.415 / 1.211× | 759.143 / 1.213× | 755.555 / 1.207× | 754.880 / 1.206× |
| Gaussian 32D, single query | 49 | 630.510 | 620.858–640.679 | 773.714 / 1.227× | 770.547 / 1.222× | 766.974 / 1.216× | 768.438 / 1.219× |
| Gaussian 8D, single query | 42 | 635.599 | 629.165–639.342 | 773.517 / 1.217× | 779.794 / 1.227× | 765.504 / 1.204× | 760.426 / 1.196× |
| Gaussian 32D, single query | 50 | 634.214 | 628.178–647.734 | 775.535 / 1.223× | 773.097 / 1.219× | 773.481 / 1.220× | 777.808 / 1.226× |
| Gaussian 8D, single query | 43 | 637.608 | 629.417–654.786 | 771.537 / 1.210× | 782.536 / 1.227× | 770.258 / 1.208× | 771.933 / 1.211× |
| Gaussian 32D, single query | 51 | 646.563 | 633.170–663.015 | 773.019 / 1.196× | 778.812 / 1.205× | 777.467 / 1.202× | 784.992 / 1.214× |
| Digits (raw) | 151 | 641.167 | 633.915–652.177 | 782.117 / 1.220× | 782.373 / 1.220× | 776.647 / 1.211× | 780.952 / 1.218× |
| Digits (raw) | 251 | 634.565 | 631.106–641.742 | 773.721 / 1.219× | 775.251 / 1.222× | 779.566 / 1.229× | 767.209 / 1.209× |
| Digits (raw) | 252 | 634.724 | 629.872–640.139 | 776.768 / 1.224× | 784.025 / 1.235× | 779.822 / 1.229× | 775.013 / 1.221× |
| Breast Cancer (raw) | 151 | 634.626 | 630.775–654.110 | 779.323 / 1.228× | 781.002 / 1.231× | 775.916 / 1.223× | 774.493 / 1.220× |
| Breast Cancer (raw) | 251 | 635.043 | 633.154–643.962 | 774.345 / 1.219× | 780.121 / 1.228× | 773.096 / 1.217× | 775.315 / 1.221× |
| Breast Cancer (raw) | 252 | 635.971 | 629.972–645.510 | 775.563 / 1.219× | 780.478 / 1.227× | 774.299 / 1.218× | 767.052 / 1.206× |
| Iris (raw) | 151 | 635.302 | 632.451–642.923 | 774.324 / 1.219× | 781.500 / 1.230× | 766.750 / 1.207× | 773.148 / 1.217× |
| Iris (raw) | 251 | 638.538 | 633.071–639.629 | 774.610 / 1.213× | 785.754 / 1.231× | 764.500 / 1.197× | 780.429 / 1.222× |
| Iris (raw) | 252 | 640.225 | 634.329–652.022 | 775.683 / 1.212× | 774.344 / 1.209× | 761.969 / 1.190× | 768.536 / 1.200× |
| Wine (raw) | 151 | 636.059 | 632.185–644.976 | 772.331 / 1.214× | 783.583 / 1.232× | 762.935 / 1.199× | 772.828 / 1.215× |
| Wine (raw) | 251 | 635.608 | 630.202–640.302 | 778.121 / 1.224× | 781.906 / 1.230× | 764.487 / 1.203× | 767.590 / 1.208× |
| Wine (raw) | 252 | 638.816 | 633.707–640.658 | 784.695 / 1.228× | 781.973 / 1.224× | 769.000 / 1.204× | 776.126 / 1.215× |

## Process start through first result

Parent-observed interpreter startup through the first result marker: includes NumPy import, fixture X/Q loading, estimator setup, fit/query and IPC; excludes the later audit and metadata.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 676.224 | 670.862–692.045 | 807.983 / 1.195× | 809.031 / 1.196× | 804.720 / 1.190× | 804.378 / 1.190× |
| Gaussian 32D, single query | 49 | 681.121 | 671.129–690.985 | 823.478 / 1.209× | 821.719 / 1.206× | 816.895 / 1.199× | 819.587 / 1.203× |
| Gaussian 8D, single query | 42 | 686.180 | 679.840–689.691 | 823.419 / 1.200× | 831.699 / 1.212× | 818.119 / 1.192× | 810.639 / 1.181× |
| Gaussian 32D, single query | 50 | 684.066 | 679.171–699.484 | 826.504 / 1.208× | 824.735 / 1.206× | 823.550 / 1.204× | 829.878 / 1.213× |
| Gaussian 8D, single query | 43 | 689.008 | 680.155–708.226 | 821.386 / 1.192× | 834.373 / 1.211× | 822.372 / 1.194× | 822.942 / 1.194× |
| Gaussian 32D, single query | 51 | 701.093 | 683.431–714.699 | 823.930 / 1.175× | 830.568 / 1.185× | 828.203 / 1.181× | 838.379 / 1.196× |
| Digits (raw) | 151 | 691.315 | 684.716–706.614 | 834.032 / 1.206× | 834.976 / 1.208× | 829.694 / 1.200× | 832.438 / 1.204× |
| Digits (raw) | 251 | 684.969 | 682.010–692.659 | 824.406 / 1.204× | 826.684 / 1.207× | 830.468 / 1.212× | 818.243 / 1.195× |
| Digits (raw) | 252 | 684.983 | 679.991–692.107 | 827.507 / 1.208× | 836.244 / 1.221× | 830.060 / 1.212× | 826.974 / 1.207× |
| Breast Cancer (raw) | 151 | 686.438 | 681.162–704.620 | 830.585 / 1.210× | 832.735 / 1.213× | 826.578 / 1.204× | 825.248 / 1.202× |
| Breast Cancer (raw) | 251 | 685.413 | 683.933–697.159 | 825.140 / 1.204× | 832.125 / 1.214× | 823.733 / 1.202× | 826.437 / 1.206× |
| Breast Cancer (raw) | 252 | 686.392 | 679.595–695.772 | 826.513 / 1.204× | 831.567 / 1.212× | 825.324 / 1.202× | 817.811 / 1.191× |
| Iris (raw) | 151 | 685.877 | 682.441–696.938 | 824.068 / 1.201× | 833.305 / 1.215× | 817.382 / 1.192× | 823.530 / 1.201× |
| Iris (raw) | 251 | 689.147 | 683.686–692.558 | 825.739 / 1.198× | 838.064 / 1.216× | 814.992 / 1.183× | 831.412 / 1.206× |
| Iris (raw) | 252 | 690.858 | 684.600–704.332 | 825.515 / 1.195× | 825.741 / 1.195× | 812.446 / 1.176× | 819.715 / 1.187× |
| Wine (raw) | 151 | 686.491 | 682.564–696.672 | 822.618 / 1.198× | 835.119 / 1.217× | 813.398 / 1.185× | 823.980 / 1.200× |
| Wine (raw) | 251 | 686.155 | 681.024–691.528 | 829.282 / 1.209× | 832.716 / 1.214× | 814.982 / 1.188× | 817.942 / 1.192× |
| Wine (raw) | 252 | 690.631 | 684.136–692.698 | 835.337 / 1.210× | 834.221 / 1.208× | 819.911 / 1.187× | 826.286 / 1.196× |

## Every slower or missing median

Every below-1 or missing median from the five phase tables is repeated here. Paired ratios compare the corresponding serial launch rounds; their median need not equal the ratio of separate medians and is not a confidence interval.

| Phase | Comparator | Case | Current ms | Comparator ms | Median ratio | Paired ratio |
| --- | --- | --- | --- | --- | --- | --- |
| First query | Preceding automatic | `shape_single_32_k5_seed50` | 0.139 | 0.136 | **0.975×** | 1.045× |
| First query | Preceding automatic | `real_digits_raw_k5_seed151` | 0.614 | 0.587 | **0.956×** | **0.965×** |
| First query | Preceding automatic | `real_digits_raw_k5_seed251` | 0.601 | 0.601 | **1.000×** | 1.004× |
| First query | Preceding automatic | `real_digits_raw_k5_seed252` | 0.594 | 0.585 | **0.985×** | **0.985×** |
| First fit + query | Preceding automatic | `shape_single_32_k5_seed51` | 0.375 | 0.363 | **0.969×** | **0.984×** |

## Audits, actual routing and native initialization

| Configuration | Launches | Valid timings | Strict passes | Errors | Query backends | Native already loaded before fit | Stock sklearn.neighbors loaded before fit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current automatic | 90 | 90 | 90 | 0 | {"cpu": 90} | {"True": 90} | {"False": 90} |
| Default-thread brute | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |
| Single-thread brute | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |
| sklearn auto | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |
| Preceding automatic | 90 | 90 | 90 | 0 | {"cpu": 90} | {"True": 90} | {"True": 90} |

A true native-loaded-before-fit flag means normal estimator imports/setup loaded the extension. The setup and total-latency tables retain this cost. The harness performs no native prewarm. Each raw observation also records module presence before NumPy input preparation and after it. Stock sklearn.neighbors presence is recorded separately from the top-level sklearn package; older evidence lacking that field remains labelled not_recorded.

### Every failed observation

| Configuration | Case | Launch | Failure | Raw JSON pointer |
| --- | --- | --- | --- | --- |
| Default-thread brute | `real_digits_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/0` |
| Default-thread brute | `real_digits_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/1` |
| Default-thread brute | `real_digits_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/2` |
| Default-thread brute | `real_digits_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/3` |
| Default-thread brute | `real_digits_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/4` |
| Single-thread brute | `real_digits_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/0` |
| Single-thread brute | `real_digits_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/1` |
| Single-thread brute | `real_digits_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/2` |
| Single-thread brute | `real_digits_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/3` |
| Single-thread brute | `real_digits_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/4` |
| sklearn auto | `real_digits_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_auto/samples/0` |
| sklearn auto | `real_digits_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_auto/samples/1` |
| sklearn auto | `real_digits_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_auto/samples/2` |
| sklearn auto | `real_digits_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_auto/samples/3` |
| sklearn auto | `real_digits_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_auto/samples/4` |
| Default-thread brute | `real_digits_raw_k5_seed251` | 1 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_default/samples/0` |
| Default-thread brute | `real_digits_raw_k5_seed251` | 2 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_default/samples/1` |
| Default-thread brute | `real_digits_raw_k5_seed251` | 3 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_default/samples/2` |
| Default-thread brute | `real_digits_raw_k5_seed251` | 4 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_default/samples/3` |
| Default-thread brute | `real_digits_raw_k5_seed251` | 5 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_default/samples/4` |
| Single-thread brute | `real_digits_raw_k5_seed251` | 1 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/0` |
| Single-thread brute | `real_digits_raw_k5_seed251` | 2 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/1` |
| Single-thread brute | `real_digits_raw_k5_seed251` | 3 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/2` |
| Single-thread brute | `real_digits_raw_k5_seed251` | 4 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/3` |
| Single-thread brute | `real_digits_raw_k5_seed251` | 5 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/4` |
| sklearn auto | `real_digits_raw_k5_seed251` | 1 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_auto/samples/0` |
| sklearn auto | `real_digits_raw_k5_seed251` | 2 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_auto/samples/1` |
| sklearn auto | `real_digits_raw_k5_seed251` | 3 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_auto/samples/2` |
| sklearn auto | `real_digits_raw_k5_seed251` | 4 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_auto/samples/3` |
| sklearn auto | `real_digits_raw_k5_seed251` | 5 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_auto/samples/4` |
| Default-thread brute | `real_digits_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_default/samples/0` |
| Default-thread brute | `real_digits_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_default/samples/1` |
| Default-thread brute | `real_digits_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_default/samples/2` |
| Default-thread brute | `real_digits_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_default/samples/3` |
| Default-thread brute | `real_digits_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_default/samples/4` |
| Single-thread brute | `real_digits_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/0` |
| Single-thread brute | `real_digits_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/1` |
| Single-thread brute | `real_digits_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/2` |
| Single-thread brute | `real_digits_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/3` |
| Single-thread brute | `real_digits_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/4` |
| sklearn auto | `real_digits_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_auto/samples/0` |
| sklearn auto | `real_digits_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_auto/samples/1` |
| sklearn auto | `real_digits_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_auto/samples/2` |
| sklearn auto | `real_digits_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_auto/samples/3` |
| sklearn auto | `real_digits_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_auto/samples/4` |
| Default-thread brute | `real_iris_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_default/samples/0` |
| Default-thread brute | `real_iris_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_default/samples/1` |
| Default-thread brute | `real_iris_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_default/samples/2` |
| Default-thread brute | `real_iris_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_default/samples/3` |
| Default-thread brute | `real_iris_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_default/samples/4` |
| Single-thread brute | `real_iris_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/0` |
| Single-thread brute | `real_iris_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/1` |
| Single-thread brute | `real_iris_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/2` |
| Single-thread brute | `real_iris_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/3` |
| Single-thread brute | `real_iris_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/4` |
| sklearn auto | `real_iris_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_auto/samples/0` |
| sklearn auto | `real_iris_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_auto/samples/1` |
| sklearn auto | `real_iris_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_auto/samples/2` |
| sklearn auto | `real_iris_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_auto/samples/3` |
| sklearn auto | `real_iris_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_auto/samples/4` |
| Default-thread brute | `real_iris_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_default/samples/0` |
| Default-thread brute | `real_iris_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_default/samples/1` |
| Default-thread brute | `real_iris_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_default/samples/2` |
| Default-thread brute | `real_iris_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_default/samples/3` |
| Default-thread brute | `real_iris_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_default/samples/4` |
| Single-thread brute | `real_iris_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/0` |
| Single-thread brute | `real_iris_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/1` |
| Single-thread brute | `real_iris_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/2` |
| Single-thread brute | `real_iris_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/3` |
| Single-thread brute | `real_iris_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/4` |
| sklearn auto | `real_iris_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/0` |
| sklearn auto | `real_iris_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/1` |
| sklearn auto | `real_iris_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/2` |
| sklearn auto | `real_iris_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/3` |
| sklearn auto | `real_iris_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/4` |

## Input preparation, fit return and post-result process lifetime

These columns keep the remaining recorded phases visible. Fit return may precede deferred preparation charged to first query. Full process elapsed includes audits, metadata and process exit after the result marker, so it is not an inference-latency measure.

| Case | Configuration | NumPy import ms | Fixture load ms | First fit ms | Worker entry→result ms | Full process elapsed ms |
| --- | --- | --- | --- | --- | --- | --- |
| `shape_single_8_k5_seed41` | Current automatic | 24.487 | 1.703 | 0.186 | 653.960 | 743.232 |
| `shape_single_8_k5_seed41` | Default-thread brute | 24.127 | 1.643 | 0.315 | 786.180 | 893.479 |
| `shape_single_8_k5_seed41` | Single-thread brute | 24.236 | 1.693 | 0.328 | 786.906 | 882.321 |
| `shape_single_8_k5_seed41` | sklearn auto | 23.965 | 1.621 | 1.707 | 782.946 | 880.604 |
| `shape_single_8_k5_seed41` | Preceding automatic | 24.123 | 1.545 | 0.222 | 782.529 | 881.470 |
| `shape_single_32_k5_seed49` | Current automatic | 24.401 | 1.748 | 0.214 | 659.034 | 748.837 |
| `shape_single_32_k5_seed49` | Default-thread brute | 24.131 | 1.710 | 0.342 | 801.637 | 908.730 |
| `shape_single_32_k5_seed49` | Single-thread brute | 24.784 | 1.792 | 0.345 | 798.975 | 897.558 |
| `shape_single_32_k5_seed49` | sklearn auto | 24.144 | 1.743 | 0.331 | 794.723 | 903.508 |
| `shape_single_32_k5_seed49` | Preceding automatic | 24.742 | 1.783 | 0.222 | 796.757 | 897.982 |
| `shape_single_8_k5_seed42` | Current automatic | 24.482 | 1.638 | 0.175 | 663.709 | 753.769 |
| `shape_single_8_k5_seed42` | Default-thread brute | 24.390 | 1.660 | 0.312 | 801.177 | 908.659 |
| `shape_single_8_k5_seed42` | Single-thread brute | 24.910 | 1.670 | 0.326 | 808.426 | 905.741 |
| `shape_single_8_k5_seed42` | sklearn auto | 24.655 | 1.630 | 1.776 | 796.307 | 895.477 |
| `shape_single_8_k5_seed42` | Preceding automatic | 24.040 | 1.611 | 0.214 | 788.147 | 898.094 |
| `shape_single_32_k5_seed50` | Current automatic | 25.013 | 1.801 | 0.223 | 662.265 | 751.411 |
| `shape_single_32_k5_seed50` | Default-thread brute | 24.979 | 1.840 | 0.323 | 804.117 | 912.772 |
| `shape_single_32_k5_seed50` | Single-thread brute | 24.926 | 1.780 | 0.359 | 801.728 | 900.343 |
| `shape_single_32_k5_seed50` | sklearn auto | 24.437 | 1.801 | 0.332 | 801.461 | 910.226 |
| `shape_single_32_k5_seed50` | Preceding automatic | 25.480 | 1.864 | 0.239 | 807.166 | 908.786 |
| `shape_single_8_k5_seed43` | Current automatic | 24.948 | 1.652 | 0.174 | 666.228 | 763.160 |
| `shape_single_8_k5_seed43` | Default-thread brute | 24.619 | 1.607 | 0.313 | 798.896 | 906.808 |
| `shape_single_8_k5_seed43` | Single-thread brute | 25.064 | 1.676 | 0.328 | 811.218 | 911.868 |
| `shape_single_8_k5_seed43` | sklearn auto | 25.403 | 1.758 | 1.751 | 799.373 | 900.264 |
| `shape_single_8_k5_seed43` | Preceding automatic | 24.538 | 1.655 | 0.210 | 800.264 | 899.684 |
| `shape_single_32_k5_seed51` | Current automatic | 25.407 | 1.833 | 0.226 | 676.926 | 770.091 |
| `shape_single_32_k5_seed51` | Default-thread brute | 24.690 | 1.846 | 0.339 | 801.507 | 915.376 |
| `shape_single_32_k5_seed51` | Single-thread brute | 25.184 | 1.772 | 0.362 | 807.706 | 905.922 |
| `shape_single_32_k5_seed51` | sklearn auto | 24.863 | 1.844 | 0.342 | 805.808 | 914.754 |
| `shape_single_32_k5_seed51` | Preceding automatic | 26.155 | 1.870 | 0.216 | 814.967 | 917.782 |
| `real_digits_raw_k5_seed151` | Current automatic | 24.921 | 1.763 | 0.099 | 668.909 | 762.160 |
| `real_digits_raw_k5_seed151` | Default-thread brute | 25.085 | 1.697 | 0.319 | 810.851 | 920.919 |
| `real_digits_raw_k5_seed151` | Single-thread brute | 24.917 | 1.725 | 0.349 | 811.154 | 913.269 |
| `real_digits_raw_k5_seed151` | sklearn auto | 25.113 | 1.883 | 0.321 | 806.424 | 915.459 |
| `real_digits_raw_k5_seed151` | Preceding automatic | 24.801 | 1.728 | 0.141 | 809.423 | 910.627 |
| `real_digits_raw_k5_seed251` | Current automatic | 24.667 | 1.754 | 0.095 | 662.738 | 754.230 |
| `real_digits_raw_k5_seed251` | Default-thread brute | 24.572 | 1.705 | 0.322 | 801.916 | 909.277 |
| `real_digits_raw_k5_seed251` | Single-thread brute | 24.946 | 1.721 | 0.339 | 804.000 | 901.777 |
| `real_digits_raw_k5_seed251` | sklearn auto | 24.549 | 1.678 | 0.317 | 808.481 | 916.691 |
| `real_digits_raw_k5_seed251` | Preceding automatic | 24.784 | 1.696 | 0.138 | 795.444 | 896.445 |
| `real_digits_raw_k5_seed252` | Current automatic | 24.401 | 1.678 | 0.095 | 662.667 | 753.558 |
| `real_digits_raw_k5_seed252` | Default-thread brute | 24.479 | 1.696 | 0.313 | 804.917 | 913.160 |
| `real_digits_raw_k5_seed252` | Single-thread brute | 25.076 | 1.769 | 0.332 | 812.747 | 915.033 |
| `real_digits_raw_k5_seed252` | sklearn auto | 24.702 | 1.708 | 0.317 | 807.717 | 917.753 |
| `real_digits_raw_k5_seed252` | Preceding automatic | 24.654 | 1.739 | 0.139 | 803.575 | 903.980 |
| `real_breast_cancer_raw_k5_seed151` | Current automatic | 24.831 | 1.650 | 0.085 | 663.135 | 754.391 |
| `real_breast_cancer_raw_k5_seed151` | Default-thread brute | 24.561 | 1.664 | 0.308 | 808.021 | 917.615 |
| `real_breast_cancer_raw_k5_seed151` | Single-thread brute | 24.797 | 1.655 | 0.332 | 809.858 | 907.357 |
| `real_breast_cancer_raw_k5_seed151` | sklearn auto | 24.552 | 1.622 | 0.327 | 803.964 | 911.481 |
| `real_breast_cancer_raw_k5_seed151` | Preceding automatic | 24.790 | 1.619 | 0.126 | 802.879 | 902.908 |
| `real_breast_cancer_raw_k5_seed251` | Current automatic | 24.618 | 1.641 | 0.087 | 663.169 | 754.133 |
| `real_breast_cancer_raw_k5_seed251` | Default-thread brute | 24.549 | 1.590 | 0.306 | 802.593 | 915.102 |
| `real_breast_cancer_raw_k5_seed251` | Single-thread brute | 25.033 | 1.671 | 0.330 | 808.764 | 907.976 |
| `real_breast_cancer_raw_k5_seed251` | sklearn auto | 24.646 | 1.625 | 0.308 | 801.419 | 908.378 |
| `real_breast_cancer_raw_k5_seed251` | Preceding automatic | 24.751 | 1.630 | 0.127 | 803.754 | 904.416 |
| `real_breast_cancer_raw_k5_seed252` | Current automatic | 24.520 | 1.607 | 0.085 | 663.995 | 754.446 |
| `real_breast_cancer_raw_k5_seed252` | Default-thread brute | 24.770 | 1.643 | 0.307 | 803.922 | 911.367 |
| `real_breast_cancer_raw_k5_seed252` | Single-thread brute | 24.584 | 1.650 | 0.330 | 808.640 | 906.929 |
| `real_breast_cancer_raw_k5_seed252` | sklearn auto | 24.661 | 1.617 | 0.305 | 802.708 | 910.316 |
| `real_breast_cancer_raw_k5_seed252` | Preceding automatic | 24.817 | 1.639 | 0.125 | 795.320 | 895.292 |
| `real_iris_raw_k5_seed151` | Current automatic | 24.641 | 1.617 | 0.082 | 663.453 | 753.350 |
| `real_iris_raw_k5_seed151` | Default-thread brute | 24.675 | 1.601 | 0.306 | 802.377 | 909.148 |
| `real_iris_raw_k5_seed151` | Single-thread brute | 24.876 | 1.645 | 0.322 | 810.179 | 908.055 |
| `real_iris_raw_k5_seed151` | sklearn auto | 24.754 | 1.614 | 0.394 | 794.814 | 893.643 |
| `real_iris_raw_k5_seed151` | Preceding automatic | 24.602 | 1.612 | 0.123 | 801.266 | 900.384 |
| `real_iris_raw_k5_seed251` | Current automatic | 24.707 | 1.638 | 0.084 | 666.750 | 757.619 |
| `real_iris_raw_k5_seed251` | Default-thread brute | 24.794 | 1.688 | 0.303 | 803.022 | 913.381 |
| `real_iris_raw_k5_seed251` | Single-thread brute | 25.266 | 1.649 | 0.340 | 814.672 | 912.312 |
| `real_iris_raw_k5_seed251` | sklearn auto | 24.625 | 1.673 | 0.396 | 792.516 | 892.922 |
| `real_iris_raw_k5_seed251` | Preceding automatic | 24.742 | 1.650 | 0.125 | 808.928 | 909.114 |
| `real_iris_raw_k5_seed252` | Current automatic | 24.442 | 1.568 | 0.084 | 668.182 | 759.440 |
| `real_iris_raw_k5_seed252` | Default-thread brute | 24.448 | 1.609 | 0.305 | 803.534 | 911.106 |
| `real_iris_raw_k5_seed252` | Single-thread brute | 24.858 | 1.578 | 0.330 | 802.766 | 900.184 |
| `real_iris_raw_k5_seed252` | sklearn auto | 24.645 | 1.627 | 0.395 | 790.136 | 887.485 |
| `real_iris_raw_k5_seed252` | Preceding automatic | 24.568 | 1.589 | 0.124 | 796.904 | 897.464 |
| `real_wine_raw_k5_seed151` | Current automatic | 24.673 | 1.615 | 0.083 | 664.204 | 755.632 |
| `real_wine_raw_k5_seed151` | Default-thread brute | 24.428 | 1.586 | 0.303 | 799.994 | 908.351 |
| `real_wine_raw_k5_seed151` | Single-thread brute | 24.899 | 1.633 | 0.324 | 812.215 | 909.658 |
| `real_wine_raw_k5_seed151` | sklearn auto | 24.652 | 1.600 | 0.397 | 791.110 | 890.465 |
| `real_wine_raw_k5_seed151` | Preceding automatic | 24.702 | 1.614 | 0.133 | 801.093 | 901.481 |
| `real_wine_raw_k5_seed251` | Current automatic | 24.716 | 1.616 | 0.082 | 663.614 | 754.541 |
| `real_wine_raw_k5_seed251` | Default-thread brute | 24.684 | 1.601 | 0.314 | 806.415 | 914.145 |
| `real_wine_raw_k5_seed251` | Single-thread brute | 24.772 | 1.638 | 0.322 | 810.451 | 908.747 |
| `real_wine_raw_k5_seed251` | sklearn auto | 24.482 | 1.612 | 0.396 | 792.693 | 891.582 |
| `real_wine_raw_k5_seed251` | Preceding automatic | 24.653 | 1.613 | 0.124 | 795.751 | 895.363 |
| `real_wine_raw_k5_seed252` | Current automatic | 24.751 | 1.661 | 0.085 | 667.761 | 761.203 |
| `real_wine_raw_k5_seed252` | Default-thread brute | 24.758 | 1.641 | 0.303 | 812.703 | 920.535 |
| `real_wine_raw_k5_seed252` | Single-thread brute | 24.695 | 1.657 | 0.329 | 810.825 | 909.849 |
| `real_wine_raw_k5_seed252` | sklearn auto | 24.745 | 1.638 | 0.398 | 797.492 | 897.252 |
| `real_wine_raw_k5_seed252` | Preceding automatic | 24.682 | 1.596 | 0.123 | 804.162 | 903.180 |

## Evidence

[Every raw launch, audit and runtime identity](../benchmarks/results/knn-startup-final2-2026-10-30-cold.json)

[Validated derived summary](../benchmarks/results/knn-startup-final2-2026-10-30-cold-context-summary.json)
