# Fresh-process KNN first-use benchmark

Completed `2026-10-31T01:21:02.860896+00:00`. This report retains **450 separate process launches** over 18 fixtures, 5 configurations and 5 launches per fixture/configuration. Accuracy/integrity acceptance: **True**.

Each retained launch starts a new interpreter and performs exactly one first public NearestNeighbors fit/query. There are no warmups, metadata preflight requests, harness-driven native preimports or external pre-fit MLX synchronization. NumPy import and fixture X/Q loading are reported separately from estimator import/setup and API calls. Necessary transitive library imports remain charged. The first stdout result marker follows returned NumPy neighbors, before audits and runtime metadata. Parent process-start-to-result includes interpreter startup, fixture input loading, estimator imports, fit/query and result-marker IPC. Process elapsed also includes post-result audits/metadata/exit. This is process-cold, not a claim of cold OS/filesystem/Metal disk caches. All launches are serial.

Fit/query reductions may reflect moving native initialization into normal estimator imports. Estimator setup and estimator/process-to-result measures retain that cost. Parent result-marker latency includes interpreter/input loading/IPC; worker estimator-to-result excludes input preparation. No launch or failed audit is removed, and no OS/bytecode/Metal-disk-cache coldness is claimed.

All times are milliseconds. Speedup is comparator time / current automatic time; a value below 1× means current is slower. Bold ratios mark every slower or missing median. No failed numerical audit removes a timing from the comparison. These are descriptive process-cold measurements, not a universal first-use latency guarantee or a machine-cold test.

## Coverage by separately measured phase

| Phase | Comparator | Current medians faster | Minimum speedup | Maximum speedup |
| --- | --- | --- | --- | --- |
| First query | Default-thread brute | 18/18 | 24.098× | 175.835× |
| First query | Single-thread brute | 18/18 | 10.810× | 76.811× |
| First query | sklearn auto | 18/18 | 3.258× | 94.512× |
| First query | Preceding automatic | 11/18 | **0.959×** | 1.078× |
| First fit + query | Default-thread brute | 18/18 | 21.216× | 86.030× |
| First fit + query | Single-thread brute | 18/18 | 9.824× | 39.245× |
| First fit + query | sklearn auto | 18/18 | 4.079× | 42.476× |
| First fit + query | Preceding automatic | 13/18 | **0.982×** | 1.075× |
| Estimator import and setup | Default-thread brute | 18/18 | 1.183× | 1.218× |
| Estimator import and setup | Single-thread brute | 18/18 | 1.187× | 1.225× |
| Estimator import and setup | sklearn auto | 18/18 | 1.169× | 1.219× |
| Estimator import and setup | Preceding automatic | 11/18 | **0.986×** | 1.017× |
| Estimator import through first result | Default-thread brute | 18/18 | 1.204× | 1.239× |
| Estimator import through first result | Single-thread brute | 18/18 | 1.196× | 1.234× |
| Estimator import through first result | sklearn auto | 18/18 | 1.172× | 1.241× |
| Estimator import through first result | Preceding automatic | 11/18 | **0.986×** | 1.017× |
| Process start through first result | Default-thread brute | 18/18 | 1.191× | 1.224× |
| Process start through first result | Single-thread brute | 18/18 | 1.181× | 1.222× |
| Process start through first result | sklearn auto | 18/18 | 1.159× | 1.221× |
| Process start through first result | Preceding automatic | 10/18 | **0.985×** | 1.020× |

## First query

First public query after the newly constructed estimator's first fit.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.083 | 0.080–0.091 | 13.137 / 157.571× | 5.762 / 69.112× | 0.291 / 3.492× | 0.086 / 1.036× |
| Gaussian 32D, single query | 49 | 0.139 | 0.129–0.148 | 13.055 / 93.782× | 5.795 / 41.631× | 13.104 / 94.129× | 0.137 / **0.987×** |
| Gaussian 8D, single query | 42 | 0.082 | 0.080–0.128 | 13.473 / 164.392× | 5.867 / 71.582× | 0.305 / 3.719× | 0.083 / 1.013× |
| Gaussian 32D, single query | 50 | 0.138 | 0.129–0.142 | 13.830 / 100.311× | 5.563 / 40.345× | 13.031 / 94.512× | 0.132 / **0.959×** |
| Gaussian 8D, single query | 43 | 0.087 | 0.079–0.093 | 13.286 / 152.415× | 5.983 / 68.642× | 0.284 / 3.258× | 0.086 / **0.981×** |
| Gaussian 32D, single query | 51 | 0.146 | 0.138–0.152 | 13.266 / 91.043× | 5.773 / 39.623× | 13.557 / 93.044× | 0.150 / 1.028× |
| Digits (raw) | 151 | 0.597 | 0.571–0.599 | 14.374 / 24.098× | 6.448 / 10.810× | 14.434 / 24.197× | 0.606 / 1.016× |
| Digits (raw) | 251 | 0.582 | 0.562–0.671 | 14.178 / 24.382× | 6.942 / 11.939× | 14.436 / 24.825× | 0.612 / 1.052× |
| Digits (raw) | 252 | 0.557 | 0.535–0.597 | 14.561 / 26.138× | 7.129 / 12.798× | 14.959 / 26.852× | 0.600 / 1.078× |
| Breast Cancer (raw) | 151 | 0.239 | 0.223–0.255 | 13.768 / 57.508× | 5.982 / 24.985× | 13.214 / 55.195× | 0.246 / 1.027× |
| Breast Cancer (raw) | 251 | 0.242 | 0.235–0.253 | 13.873 / 57.386× | 5.847 / 24.184× | 13.679 / 56.583× | 0.248 / 1.025× |
| Breast Cancer (raw) | 252 | 0.239 | 0.234–0.261 | 14.475 / 60.650× | 5.759 / 24.129× | 13.560 / 56.816× | 0.236 / **0.989×** |
| Iris (raw) | 151 | 0.076 | 0.075–0.093 | 13.371 / 175.835× | 5.639 / 74.156× | 0.294 / 3.864× | 0.077 / 1.013× |
| Iris (raw) | 251 | 0.077 | 0.073–0.087 | 13.302 / 173.784× | 5.879 / 76.811× | 0.291 / 3.805× | 0.076 / **0.996×** |
| Iris (raw) | 252 | 0.076 | 0.075–0.082 | 12.819 / 169.411× | 5.512 / 72.843× | 0.277 / 3.656× | 0.074 / **0.984×** |
| Wine (raw) | 151 | 0.089 | 0.086–0.092 | 13.962 / 156.070× | 5.888 / 65.815× | 0.308 / 3.438× | 0.091 / 1.022× |
| Wine (raw) | 251 | 0.087 | 0.084–0.111 | 13.268 / 152.579× | 5.646 / 64.923× | 0.294 / 3.385× | 0.086 / **0.993×** |
| Wine (raw) | 252 | 0.084 | 0.077–0.090 | 12.780 / 152.451× | 5.512 / 65.750× | 0.293 / 3.500× | 0.086 / 1.026× |

## First fit + query

Constructor, first fit and first query. Estimator imports/setup are separate.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.257 | 0.248–0.268 | 13.440 / 52.390× | 6.090 / 23.741× | 2.032 / 7.922× | 0.276 / 1.075× |
| Gaussian 32D, single query | 49 | 0.357 | 0.321–0.374 | 13.379 / 37.449× | 6.159 / 17.240× | 13.441 / 37.622× | 0.360 / 1.007× |
| Gaussian 8D, single query | 42 | 0.256 | 0.250–0.305 | 13.785 / 53.927× | 6.192 / 24.224× | 2.045 / 7.998× | 0.258 / 1.011× |
| Gaussian 32D, single query | 50 | 0.356 | 0.342–0.363 | 14.149 / 39.764× | 5.940 / 16.694× | 13.353 / 37.527× | 0.359 / 1.010× |
| Gaussian 8D, single query | 43 | 0.259 | 0.243–0.269 | 13.607 / 52.620× | 6.325 / 24.460× | 2.018 / 7.802× | 0.270 / 1.045× |
| Gaussian 32D, single query | 51 | 0.356 | 0.354–0.385 | 13.604 / 38.195× | 6.131 / 17.214× | 13.891 / 39.003× | 0.381 / 1.071× |
| Digits (raw) | 151 | 0.693 | 0.660–0.697 | 14.694 / 21.216× | 6.804 / 9.824× | 14.776 / 21.334× | 0.702 / 1.013× |
| Digits (raw) | 251 | 0.681 | 0.658–0.765 | 14.492 / 21.268× | 7.278 / 10.681× | 14.771 / 21.677× | 0.706 / 1.035× |
| Digits (raw) | 252 | 0.647 | 0.624–0.695 | 14.880 / 22.984× | 7.473 / 11.543× | 15.278 / 23.599× | 0.695 / 1.073× |
| Breast Cancer (raw) | 151 | 0.327 | 0.310–0.343 | 14.069 / 43.032× | 6.324 / 19.342× | 13.521 / 41.354× | 0.337 / 1.032× |
| Breast Cancer (raw) | 251 | 0.333 | 0.321–0.337 | 14.211 / 42.680× | 6.169 / 18.529× | 13.977 / 41.978× | 0.333 / 1.001× |
| Breast Cancer (raw) | 252 | 0.327 | 0.320–0.376 | 14.786 / 45.207× | 6.106 / 18.668× | 13.893 / 42.476× | 0.325 / **0.993×** |
| Iris (raw) | 151 | 0.160 | 0.159–0.177 | 13.675 / 85.645× | 5.967 / 37.374× | 0.686 / 4.297× | 0.159 / **0.996×** |
| Iris (raw) | 251 | 0.158 | 0.151–0.176 | 13.600 / 86.030× | 6.204 / 39.245× | 0.688 / 4.352× | 0.157 / **0.991×** |
| Iris (raw) | 252 | 0.160 | 0.154–0.180 | 13.115 / 82.201× | 5.834 / 36.565× | 0.656 / 4.113× | 0.157 / **0.982×** |
| Wine (raw) | 151 | 0.175 | 0.173–0.180 | 14.266 / 81.639× | 6.221 / 35.599× | 0.716 / 4.096× | 0.176 / 1.008× |
| Wine (raw) | 251 | 0.170 | 0.164–0.197 | 13.569 / 79.800× | 5.994 / 35.249× | 0.694 / 4.079× | 0.168 / **0.989×** |
| Wine (raw) | 252 | 0.169 | 0.160–0.175 | 13.077 / 77.532× | 5.866 / 34.778× | 0.696 / 4.129× | 0.171 / 1.013× |

## Estimator import and setup

Normal estimator dependencies and configuration setup, including native initialization loaded by those imports.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 634.827 | 618.662–637.163 | 751.047 / 1.183× | 753.370 / 1.187× | 742.254 / 1.169× | 630.396 / **0.993×** |
| Gaussian 32D, single query | 49 | 627.513 | 618.046–629.782 | 751.037 / 1.197× | 765.551 / 1.220× | 751.123 / 1.197× | 630.625 / 1.005× |
| Gaussian 8D, single query | 42 | 629.631 | 626.040–630.409 | 754.590 / 1.198× | 768.517 / 1.221× | 762.299 / 1.211× | 635.294 / 1.009× |
| Gaussian 32D, single query | 50 | 632.040 | 628.338–637.904 | 765.253 / 1.211× | 764.569 / 1.210× | 755.572 / 1.195× | 634.346 / 1.004× |
| Gaussian 8D, single query | 43 | 637.596 | 623.632–642.575 | 760.853 / 1.193× | 772.496 / 1.212× | 755.766 / 1.185× | 628.489 / **0.986×** |
| Gaussian 32D, single query | 51 | 632.135 | 629.596–652.573 | 764.573 / 1.210× | 769.035 / 1.217× | 753.781 / 1.192× | 641.489 / 1.015× |
| Digits (raw) | 151 | 639.630 | 631.822–657.470 | 766.143 / 1.198× | 782.129 / 1.223× | 759.194 / 1.187× | 641.922 / 1.004× |
| Digits (raw) | 251 | 635.977 | 632.348–656.805 | 774.448 / 1.218× | 776.702 / 1.221× | 775.128 / 1.219× | 640.068 / 1.006× |
| Digits (raw) | 252 | 655.019 | 635.949–660.350 | 778.996 / 1.189× | 795.077 / 1.214× | 775.760 / 1.184× | 649.915 / **0.992×** |
| Breast Cancer (raw) | 151 | 653.707 | 650.781–674.553 | 779.456 / 1.192× | 800.360 / 1.224× | 784.880 / 1.201× | 656.119 / 1.004× |
| Breast Cancer (raw) | 251 | 645.432 | 634.082–678.128 | 786.412 / 1.218× | 787.299 / 1.220× | 781.008 / 1.210× | 650.854 / 1.008× |
| Breast Cancer (raw) | 252 | 651.949 | 630.304–681.447 | 775.988 / 1.190× | 796.215 / 1.221× | 777.460 / 1.193× | 644.730 / **0.989×** |
| Iris (raw) | 151 | 638.663 | 629.937–653.375 | 770.145 / 1.206× | 782.555 / 1.225× | 771.383 / 1.208× | 637.289 / **0.998×** |
| Iris (raw) | 251 | 640.034 | 632.890–643.438 | 763.748 / 1.193× | 774.365 / 1.210× | 760.148 / 1.188× | 642.422 / 1.004× |
| Iris (raw) | 252 | 647.014 | 625.887–668.059 | 770.205 / 1.190× | 773.119 / 1.195× | 761.365 / 1.177× | 642.336 / **0.993×** |
| Wine (raw) | 151 | 647.035 | 640.957–679.733 | 779.775 / 1.205× | 786.983 / 1.216× | 767.752 / 1.187× | 658.307 / 1.017× |
| Wine (raw) | 251 | 636.821 | 633.042–642.332 | 770.702 / 1.210× | 776.687 / 1.220× | 756.240 / 1.188× | 636.807 / **1.000×** |
| Wine (raw) | 252 | 633.958 | 626.953–649.949 | 754.682 / 1.190× | 768.145 / 1.212× | 758.654 / 1.197× | 635.641 / 1.003× |

## Estimator import through first result

Estimator import/setup plus first fit/query. Moving native initialization into imports does not remove it from this measure.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 635.076 | 618.911–637.430 | 764.464 / 1.204× | 759.460 / 1.196× | 744.286 / 1.172× | 630.672 / **0.993×** |
| Gaussian 32D, single query | 49 | 627.870 | 618.366–630.134 | 764.415 / 1.217× | 771.827 / 1.229× | 764.211 / 1.217× | 630.986 / 1.005× |
| Gaussian 8D, single query | 42 | 629.886 | 626.296–630.714 | 768.595 / 1.220× | 774.058 / 1.229× | 764.235 / 1.213× | 635.560 / 1.009× |
| Gaussian 32D, single query | 50 | 632.384 | 628.680–638.260 | 778.869 / 1.232× | 770.479 / 1.218× | 768.808 / 1.216× | 634.694 / 1.004× |
| Gaussian 8D, single query | 43 | 637.839 | 623.888–642.842 | 774.460 / 1.214× | 779.103 / 1.221× | 757.783 / 1.188× | 628.756 / **0.986×** |
| Gaussian 32D, single query | 51 | 632.490 | 629.952–652.942 | 778.311 / 1.231× | 775.557 / 1.226× | 767.673 / 1.214× | 641.870 / 1.015× |
| Digits (raw) | 151 | 640.290 | 632.514–658.167 | 781.053 / 1.220× | 788.860 / 1.232× | 773.971 / 1.209× | 642.602 / 1.004× |
| Digits (raw) | 251 | 636.636 | 633.029–657.509 | 788.684 / 1.239× | 784.024 / 1.232× | 790.090 / 1.241× | 640.745 / 1.006× |
| Digits (raw) | 252 | 655.667 | 636.594–661.032 | 793.506 / 1.210× | 802.550 / 1.224× | 792.147 / 1.208× | 650.610 / **0.992×** |
| Breast Cancer (raw) | 151 | 654.034 | 651.124–674.886 | 793.548 / 1.213× | 806.668 / 1.233× | 798.401 / 1.221× | 656.444 / 1.004× |
| Breast Cancer (raw) | 251 | 645.769 | 634.407–678.463 | 800.122 / 1.239× | 793.685 / 1.229× | 794.985 / 1.231× | 651.186 / 1.008× |
| Breast Cancer (raw) | 252 | 652.283 | 630.629–681.823 | 791.726 / 1.214× | 802.886 / 1.231× | 791.354 / 1.213× | 645.060 / **0.989×** |
| Iris (raw) | 151 | 638.822 | 630.097–653.552 | 783.169 / 1.226× | 788.468 / 1.234× | 772.063 / 1.209× | 637.446 / **0.998×** |
| Iris (raw) | 251 | 640.184 | 633.066–643.592 | 777.348 / 1.214× | 779.866 / 1.218× | 760.836 / 1.188× | 642.564 / 1.004× |
| Iris (raw) | 252 | 647.175 | 626.044–668.239 | 783.199 / 1.210× | 778.953 / 1.204× | 761.991 / 1.177× | 642.493 / **0.993×** |
| Wine (raw) | 151 | 647.210 | 641.132–679.913 | 795.594 / 1.229× | 793.919 / 1.227× | 768.475 / 1.187× | 658.483 / 1.017× |
| Wine (raw) | 251 | 636.987 | 633.213–642.495 | 784.115 / 1.231× | 782.681 / 1.229× | 756.926 / 1.188× | 636.975 / **1.000×** |
| Wine (raw) | 252 | 634.120 | 627.122–650.124 | 768.002 / 1.211× | 774.011 / 1.221× | 759.369 / 1.198× | 635.811 / 1.003× |

## Process start through first result

Parent-observed interpreter startup through the first result marker: includes NumPy import, fixture X/Q loading, estimator setup, fit/query and IPC; excludes the later audit and metadata.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 685.088 | 668.251–688.455 | 815.635 / 1.191× | 809.375 / 1.181× | 793.790 / 1.159× | 681.102 / **0.994×** |
| Gaussian 32D, single query | 49 | 678.775 | 668.093–681.001 | 814.023 / 1.199× | 823.544 / 1.213× | 814.159 / 1.199× | 680.723 / 1.003× |
| Gaussian 8D, single query | 42 | 679.967 | 676.771–681.111 | 817.946 / 1.203× | 823.803 / 1.212× | 814.275 / 1.198× | 684.556 / 1.007× |
| Gaussian 32D, single query | 50 | 682.770 | 679.536–688.320 | 828.912 / 1.214× | 821.856 / 1.204× | 818.582 / 1.199× | 684.742 / 1.003× |
| Gaussian 8D, single query | 43 | 689.608 | 673.879–694.792 | 824.340 / 1.195× | 829.722 / 1.203× | 808.079 / 1.172× | 679.157 / **0.985×** |
| Gaussian 32D, single query | 51 | 682.906 | 680.454–703.392 | 828.915 / 1.214× | 828.123 / 1.213× | 817.574 / 1.197× | 693.953 / 1.016× |
| Digits (raw) | 151 | 691.810 | 682.270–709.969 | 831.546 / 1.202× | 840.595 / 1.215× | 824.611 / 1.192× | 694.603 / 1.004× |
| Digits (raw) | 251 | 688.186 | 684.772–709.741 | 839.027 / 1.219× | 834.852 / 1.213× | 840.568 / 1.221× | 692.267 / 1.006× |
| Digits (raw) | 252 | 708.432 | 686.801–713.068 | 846.801 / 1.195× | 857.133 / 1.210× | 843.872 / 1.191× | 706.089 / **0.997×** |
| Breast Cancer (raw) | 151 | 711.165 | 705.260–726.931 | 846.920 / 1.191× | 858.538 / 1.207× | 852.578 / 1.199× | 709.565 / **0.998×** |
| Breast Cancer (raw) | 251 | 696.484 | 684.942–734.774 | 852.711 / 1.224× | 846.922 / 1.216× | 846.900 / 1.216× | 702.921 / 1.009× |
| Breast Cancer (raw) | 252 | 703.611 | 680.549–738.005 | 845.574 / 1.202× | 859.677 / 1.222× | 844.310 / 1.200× | 697.821 / **0.992×** |
| Iris (raw) | 151 | 690.064 | 679.748–706.315 | 834.757 / 1.210× | 840.849 / 1.219× | 822.620 / 1.192× | 687.847 / **0.997×** |
| Iris (raw) | 251 | 693.800 | 683.507–695.112 | 827.754 / 1.193× | 830.962 / 1.198× | 811.274 / 1.169× | 692.618 / **0.998×** |
| Iris (raw) | 252 | 699.268 | 676.152–721.961 | 834.205 / 1.193× | 830.657 / 1.188× | 812.489 / 1.162× | 693.285 / **0.991×** |
| Wine (raw) | 151 | 697.753 | 691.835–734.842 | 849.338 / 1.217× | 846.930 / 1.214× | 820.563 / 1.176× | 711.379 / 1.020× |
| Wine (raw) | 251 | 687.616 | 684.391–694.906 | 835.538 / 1.215× | 832.128 / 1.210× | 806.965 / 1.174× | 687.810 / 1.000× |
| Wine (raw) | 252 | 684.218 | 678.033–702.027 | 817.903 / 1.195× | 824.334 / 1.205× | 809.434 / 1.183× | 685.515 / 1.002× |

## Every slower or missing median

Every below-1 or missing median from the five phase tables is repeated here. Paired ratios compare the corresponding serial launch rounds; their median need not equal the ratio of separate medians and is not a confidence interval.

| Phase | Comparator | Case | Current ms | Comparator ms | Median ratio | Paired ratio |
| --- | --- | --- | --- | --- | --- | --- |
| First query | Preceding automatic | `shape_single_32_k5_seed49` | 0.139 | 0.137 | **0.987×** | 1.001× |
| First query | Preceding automatic | `shape_single_32_k5_seed50` | 0.138 | 0.132 | **0.959×** | **0.946×** |
| First query | Preceding automatic | `shape_single_8_k5_seed43` | 0.087 | 0.086 | **0.981×** | 1.022× |
| First query | Preceding automatic | `real_breast_cancer_raw_k5_seed252` | 0.239 | 0.236 | **0.989×** | **0.989×** |
| First query | Preceding automatic | `real_iris_raw_k5_seed251` | 0.077 | 0.076 | **0.996×** | **0.917×** |
| First query | Preceding automatic | `real_iris_raw_k5_seed252` | 0.076 | 0.074 | **0.984×** | **0.984×** |
| First query | Preceding automatic | `real_wine_raw_k5_seed251` | 0.087 | 0.086 | **0.993×** | 1.022× |
| First fit + query | Preceding automatic | `real_breast_cancer_raw_k5_seed252` | 0.327 | 0.325 | **0.993×** | **0.990×** |
| First fit + query | Preceding automatic | `real_iris_raw_k5_seed151` | 0.160 | 0.159 | **0.996×** | **0.984×** |
| First fit + query | Preceding automatic | `real_iris_raw_k5_seed251` | 0.158 | 0.157 | **0.991×** | **0.983×** |
| First fit + query | Preceding automatic | `real_iris_raw_k5_seed252` | 0.160 | 0.157 | **0.982×** | **0.997×** |
| First fit + query | Preceding automatic | `real_wine_raw_k5_seed251` | 0.170 | 0.168 | **0.989×** | 1.024× |
| Estimator import and setup | Preceding automatic | `shape_single_8_k5_seed41` | 634.827 | 630.396 | **0.993×** | 1.011× |
| Estimator import and setup | Preceding automatic | `shape_single_8_k5_seed43` | 637.596 | 628.489 | **0.986×** | **0.998×** |
| Estimator import and setup | Preceding automatic | `real_digits_raw_k5_seed252` | 655.019 | 649.915 | **0.992×** | **0.994×** |
| Estimator import and setup | Preceding automatic | `real_breast_cancer_raw_k5_seed252` | 651.949 | 644.730 | **0.989×** | **0.989×** |
| Estimator import and setup | Preceding automatic | `real_iris_raw_k5_seed151` | 638.663 | 637.289 | **0.998×** | 1.005× |
| Estimator import and setup | Preceding automatic | `real_iris_raw_k5_seed252` | 647.014 | 642.336 | **0.993×** | **0.989×** |
| Estimator import and setup | Preceding automatic | `real_wine_raw_k5_seed251` | 636.821 | 636.807 | **1.000×** | 1.004× |
| Estimator import through first result | Preceding automatic | `shape_single_8_k5_seed41` | 635.076 | 630.672 | **0.993×** | 1.012× |
| Estimator import through first result | Preceding automatic | `shape_single_8_k5_seed43` | 637.839 | 628.756 | **0.986×** | **0.998×** |
| Estimator import through first result | Preceding automatic | `real_digits_raw_k5_seed252` | 655.667 | 650.610 | **0.992×** | **0.994×** |
| Estimator import through first result | Preceding automatic | `real_breast_cancer_raw_k5_seed252` | 652.283 | 645.060 | **0.989×** | **0.989×** |
| Estimator import through first result | Preceding automatic | `real_iris_raw_k5_seed151` | 638.822 | 637.446 | **0.998×** | 1.005× |
| Estimator import through first result | Preceding automatic | `real_iris_raw_k5_seed252` | 647.175 | 642.493 | **0.993×** | **0.989×** |
| Estimator import through first result | Preceding automatic | `real_wine_raw_k5_seed251` | 636.987 | 636.975 | **1.000×** | 1.004× |
| Process start through first result | Preceding automatic | `shape_single_8_k5_seed41` | 685.088 | 681.102 | **0.994×** | 1.012× |
| Process start through first result | Preceding automatic | `shape_single_8_k5_seed43` | 689.608 | 679.157 | **0.985×** | **0.997×** |
| Process start through first result | Preceding automatic | `real_digits_raw_k5_seed252` | 708.432 | 706.089 | **0.997×** | **0.997×** |
| Process start through first result | Preceding automatic | `real_breast_cancer_raw_k5_seed151` | 711.165 | 709.565 | **0.998×** | **0.983×** |
| Process start through first result | Preceding automatic | `real_breast_cancer_raw_k5_seed252` | 703.611 | 697.821 | **0.992×** | **0.992×** |
| Process start through first result | Preceding automatic | `real_iris_raw_k5_seed151` | 690.064 | 687.847 | **0.997×** | 1.004× |
| Process start through first result | Preceding automatic | `real_iris_raw_k5_seed251` | 693.800 | 692.618 | **0.998×** | **0.998×** |
| Process start through first result | Preceding automatic | `real_iris_raw_k5_seed252` | 699.268 | 693.285 | **0.991×** | **0.991×** |

## Audits, actual routing and native initialization

| Configuration | Launches | Valid timings | Strict passes | Errors | Query backends | Native already loaded before fit | Stock sklearn.neighbors loaded before fit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current automatic | 90 | 90 | 90 | 0 | {"cpu": 90} | {"True": 90} | {"False": 90} |
| Default-thread brute | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |
| Single-thread brute | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |
| sklearn auto | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |
| Preceding automatic | 90 | 90 | 90 | 0 | {"cpu": 90} | {"True": 90} | {"False": 90} |

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
| `shape_single_8_k5_seed41` | Current automatic | 24.408 | 1.585 | 0.170 | 662.426 | 752.561 |
| `shape_single_8_k5_seed41` | Default-thread brute | 24.280 | 1.658 | 0.303 | 793.718 | 900.837 |
| `shape_single_8_k5_seed41` | Single-thread brute | 24.351 | 1.626 | 0.328 | 787.301 | 882.094 |
| `shape_single_8_k5_seed41` | sklearn auto | 24.170 | 1.600 | 1.745 | 772.172 | 868.712 |
| `shape_single_8_k5_seed41` | Preceding automatic | 24.085 | 1.623 | 0.189 | 659.039 | 747.695 |
| `shape_single_32_k5_seed49` | Current automatic | 24.506 | 1.743 | 0.213 | 656.327 | 747.021 |
| `shape_single_32_k5_seed49` | Default-thread brute | 24.177 | 1.737 | 0.323 | 792.146 | 898.174 |
| `shape_single_32_k5_seed49` | Single-thread brute | 24.789 | 1.789 | 0.352 | 800.532 | 897.468 |
| `shape_single_32_k5_seed49` | sklearn auto | 24.493 | 1.752 | 0.327 | 792.246 | 898.769 |
| `shape_single_32_k5_seed49` | Preceding automatic | 24.565 | 1.756 | 0.218 | 658.912 | 750.200 |
| `shape_single_8_k5_seed42` | Current automatic | 24.584 | 1.609 | 0.174 | 658.323 | 747.416 |
| `shape_single_8_k5_seed42` | Default-thread brute | 24.455 | 1.661 | 0.312 | 796.204 | 902.149 |
| `shape_single_8_k5_seed42` | Single-thread brute | 24.522 | 1.648 | 0.326 | 802.035 | 897.347 |
| `shape_single_8_k5_seed42` | sklearn auto | 24.356 | 1.640 | 1.746 | 792.114 | 890.314 |
| `shape_single_8_k5_seed42` | Preceding automatic | 24.509 | 1.644 | 0.175 | 663.073 | 751.544 |
| `shape_single_32_k5_seed50` | Current automatic | 24.563 | 1.795 | 0.214 | 660.806 | 752.254 |
| `shape_single_32_k5_seed50` | Default-thread brute | 24.500 | 1.753 | 0.337 | 806.887 | 915.110 |
| `shape_single_32_k5_seed50` | Single-thread brute | 25.069 | 1.782 | 0.349 | 799.082 | 897.749 |
| `shape_single_32_k5_seed50` | sklearn auto | 24.466 | 1.766 | 0.335 | 796.696 | 903.501 |
| `shape_single_32_k5_seed50` | Preceding automatic | 24.503 | 1.789 | 0.229 | 662.691 | 755.190 |
| `shape_single_8_k5_seed43` | Current automatic | 24.867 | 1.665 | 0.173 | 666.523 | 755.702 |
| `shape_single_8_k5_seed43` | Default-thread brute | 24.532 | 1.644 | 0.313 | 802.183 | 908.250 |
| `shape_single_8_k5_seed43` | Single-thread brute | 24.687 | 1.702 | 0.342 | 807.336 | 905.369 |
| `shape_single_8_k5_seed43` | sklearn auto | 24.532 | 1.641 | 1.723 | 785.871 | 883.154 |
| `shape_single_8_k5_seed43` | Preceding automatic | 24.503 | 1.683 | 0.181 | 656.798 | 746.636 |
| `shape_single_32_k5_seed51` | Current automatic | 24.569 | 1.728 | 0.217 | 660.652 | 755.868 |
| `shape_single_32_k5_seed51` | Default-thread brute | 24.709 | 1.797 | 0.348 | 806.723 | 913.664 |
| `shape_single_32_k5_seed51` | Single-thread brute | 24.894 | 1.774 | 0.360 | 804.729 | 904.646 |
| `shape_single_32_k5_seed51` | sklearn auto | 24.356 | 1.755 | 0.339 | 795.510 | 904.061 |
| `shape_single_32_k5_seed51` | Preceding automatic | 25.036 | 1.826 | 0.231 | 670.671 | 763.082 |
| `real_digits_raw_k5_seed151` | Current automatic | 24.983 | 1.732 | 0.098 | 669.179 | 762.157 |
| `real_digits_raw_k5_seed151` | Default-thread brute | 24.748 | 1.739 | 0.320 | 809.552 | 915.849 |
| `real_digits_raw_k5_seed151` | Single-thread brute | 25.292 | 1.753 | 0.344 | 817.469 | 914.442 |
| `real_digits_raw_k5_seed151` | sklearn auto | 24.510 | 1.689 | 0.316 | 802.238 | 911.190 |
| `real_digits_raw_k5_seed151` | Preceding automatic | 25.229 | 1.705 | 0.096 | 671.545 | 762.774 |
| `real_digits_raw_k5_seed251` | Current automatic | 24.880 | 1.728 | 0.096 | 665.191 | 756.150 |
| `real_digits_raw_k5_seed251` | Default-thread brute | 24.703 | 1.772 | 0.319 | 816.803 | 926.900 |
| `real_digits_raw_k5_seed251` | Single-thread brute | 24.921 | 1.772 | 0.340 | 812.348 | 911.417 |
| `real_digits_raw_k5_seed251` | sklearn auto | 25.320 | 1.759 | 0.326 | 818.190 | 926.402 |
| `real_digits_raw_k5_seed251` | Preceding automatic | 24.787 | 1.703 | 0.098 | 669.189 | 761.311 |
| `real_digits_raw_k5_seed252` | Current automatic | 25.227 | 1.753 | 0.096 | 684.607 | 781.479 |
| `real_digits_raw_k5_seed252` | Default-thread brute | 25.332 | 1.731 | 0.319 | 823.262 | 939.051 |
| `real_digits_raw_k5_seed252` | Single-thread brute | 25.800 | 1.748 | 0.369 | 832.559 | 937.006 |
| `real_digits_raw_k5_seed252` | sklearn auto | 25.163 | 1.766 | 0.328 | 821.047 | 935.933 |
| `real_digits_raw_k5_seed252` | Preceding automatic | 25.922 | 1.801 | 0.099 | 681.634 | 775.150 |
| `real_breast_cancer_raw_k5_seed151` | Current automatic | 25.329 | 1.673 | 0.088 | 684.619 | 789.847 |
| `real_breast_cancer_raw_k5_seed151` | Default-thread brute | 25.976 | 1.707 | 0.309 | 823.386 | 931.643 |
| `real_breast_cancer_raw_k5_seed151` | Single-thread brute | 25.432 | 1.746 | 0.338 | 835.588 | 936.950 |
| `real_breast_cancer_raw_k5_seed151` | sklearn auto | 25.706 | 1.636 | 0.311 | 829.344 | 938.124 |
| `real_breast_cancer_raw_k5_seed151` | Preceding automatic | 25.347 | 1.667 | 0.091 | 686.632 | 780.746 |
| `real_breast_cancer_raw_k5_seed251` | Current automatic | 24.706 | 1.619 | 0.085 | 673.982 | 768.466 |
| `real_breast_cancer_raw_k5_seed251` | Default-thread brute | 25.180 | 1.686 | 0.306 | 829.259 | 936.336 |
| `real_breast_cancer_raw_k5_seed251` | Single-thread brute | 25.478 | 1.663 | 0.349 | 822.986 | 923.232 |
| `real_breast_cancer_raw_k5_seed251` | sklearn auto | 25.224 | 1.723 | 0.314 | 824.309 | 932.527 |
| `real_breast_cancer_raw_k5_seed251` | Preceding automatic | 25.224 | 1.666 | 0.090 | 680.258 | 773.637 |
| `real_breast_cancer_raw_k5_seed252` | Current automatic | 25.059 | 1.607 | 0.089 | 681.089 | 776.500 |
| `real_breast_cancer_raw_k5_seed252` | Default-thread brute | 25.686 | 1.665 | 0.324 | 821.066 | 940.644 |
| `real_breast_cancer_raw_k5_seed252` | Single-thread brute | 25.695 | 1.718 | 0.349 | 833.444 | 938.576 |
| `real_breast_cancer_raw_k5_seed252` | sklearn auto | 25.844 | 1.672 | 0.316 | 820.716 | 931.852 |
| `real_breast_cancer_raw_k5_seed252` | Preceding automatic | 25.098 | 1.670 | 0.086 | 674.240 | 766.044 |
| `real_iris_raw_k5_seed151` | Current automatic | 25.092 | 1.607 | 0.084 | 667.740 | 758.122 |
| `real_iris_raw_k5_seed151` | Default-thread brute | 24.983 | 1.617 | 0.305 | 811.473 | 919.239 |
| `real_iris_raw_k5_seed151` | Single-thread brute | 25.321 | 1.629 | 0.329 | 817.353 | 915.446 |
| `real_iris_raw_k5_seed151` | sklearn auto | 24.724 | 1.615 | 0.396 | 800.304 | 906.589 |
| `real_iris_raw_k5_seed151` | Preceding automatic | 24.992 | 1.612 | 0.082 | 665.540 | 755.765 |
| `real_iris_raw_k5_seed251` | Current automatic | 24.604 | 1.607 | 0.082 | 669.974 | 760.985 |
| `real_iris_raw_k5_seed251` | Default-thread brute | 24.629 | 1.602 | 0.298 | 805.405 | 910.092 |
| `real_iris_raw_k5_seed251` | Single-thread brute | 24.772 | 1.606 | 0.325 | 808.271 | 905.806 |
| `real_iris_raw_k5_seed251` | sklearn auto | 24.670 | 1.619 | 0.390 | 788.947 | 886.558 |
| `real_iris_raw_k5_seed251` | Preceding automatic | 24.535 | 1.607 | 0.081 | 670.443 | 761.366 |
| `real_iris_raw_k5_seed252` | Current automatic | 25.000 | 1.606 | 0.082 | 675.804 | 770.067 |
| `real_iris_raw_k5_seed252` | Default-thread brute | 24.396 | 1.572 | 0.300 | 811.818 | 918.641 |
| `real_iris_raw_k5_seed252` | Single-thread brute | 24.805 | 1.680 | 0.318 | 807.293 | 903.736 |
| `real_iris_raw_k5_seed252` | sklearn auto | 24.570 | 1.580 | 0.382 | 790.060 | 892.031 |
| `real_iris_raw_k5_seed252` | Preceding automatic | 24.616 | 1.587 | 0.082 | 670.689 | 760.379 |
| `real_wine_raw_k5_seed151` | Current automatic | 25.102 | 1.635 | 0.086 | 675.383 | 766.129 |
| `real_wine_raw_k5_seed151` | Default-thread brute | 24.495 | 1.644 | 0.308 | 825.806 | 943.201 |
| `real_wine_raw_k5_seed151` | Single-thread brute | 25.566 | 1.639 | 0.324 | 823.126 | 921.284 |
| `real_wine_raw_k5_seed151` | sklearn auto | 25.221 | 1.657 | 0.407 | 797.090 | 900.057 |
| `real_wine_raw_k5_seed151` | Preceding automatic | 25.398 | 1.817 | 0.085 | 687.962 | 781.173 |
| `real_wine_raw_k5_seed251` | Current automatic | 24.870 | 1.624 | 0.083 | 665.031 | 757.586 |
| `real_wine_raw_k5_seed251` | Default-thread brute | 24.556 | 1.635 | 0.301 | 812.970 | 922.203 |
| `real_wine_raw_k5_seed251` | Single-thread brute | 24.624 | 1.605 | 0.330 | 810.417 | 909.277 |
| `real_wine_raw_k5_seed251` | sklearn auto | 24.384 | 1.585 | 0.406 | 784.772 | 884.067 |
| `real_wine_raw_k5_seed251` | Preceding automatic | 24.715 | 1.665 | 0.082 | 665.631 | 756.578 |
| `real_wine_raw_k5_seed252` | Current automatic | 24.878 | 1.616 | 0.083 | 661.826 | 750.220 |
| `real_wine_raw_k5_seed252` | Default-thread brute | 24.585 | 1.622 | 0.296 | 795.699 | 900.463 |
| `real_wine_raw_k5_seed252` | Single-thread brute | 24.532 | 1.608 | 0.317 | 801.727 | 899.077 |
| `real_wine_raw_k5_seed252` | sklearn auto | 24.423 | 1.594 | 0.398 | 787.269 | 886.568 |
| `real_wine_raw_k5_seed252` | Preceding automatic | 24.652 | 1.622 | 0.085 | 663.454 | 752.924 |

## Evidence

[Every raw launch, audit and runtime identity](../benchmarks/results/knn-digits-final-2026-10-30-cold.json)

[Validated derived summary](../benchmarks/results/knn-digits-final-2026-10-30-cold-summary.json)
