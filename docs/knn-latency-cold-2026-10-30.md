# Fresh-process KNN first-use benchmark

Completed `2026-10-30T22:07:37.930879+00:00`. This report retains **450 separate process launches** over 18 fixtures, 5 configurations and 5 launches per fixture/configuration. Accuracy/integrity acceptance: **True**.

Each retained launch starts a new interpreter and performs exactly one first public NearestNeighbors fit/query. There are no warmups, metadata preflight requests, harness-driven native preimports or external pre-fit MLX synchronization. NumPy import and fixture X/Q loading are reported separately from estimator import/setup and API calls. Necessary transitive library imports remain charged. The first stdout result marker follows returned NumPy neighbors, before audits and runtime metadata. Parent process-start-to-result includes interpreter startup, fixture input loading, estimator imports, fit/query and result-marker IPC. Process elapsed also includes post-result audits/metadata/exit. This is process-cold, not a claim of cold OS/filesystem/Metal disk caches. All launches are serial.

Fit/query reductions may reflect moving native initialization into normal estimator imports. Estimator setup and estimator/process-to-result measures retain that cost. Parent result-marker latency includes interpreter/input loading/IPC; worker estimator-to-result excludes input preparation. No launch or failed audit is removed, and no OS/bytecode/Metal-disk-cache coldness is claimed.

All times are milliseconds. Speedup is comparator time / current automatic time; a value below 1× means current is slower. Bold ratios mark every slower or missing median. No failed numerical audit removes a timing from the comparison. These are descriptive process-cold measurements, not a universal first-use latency guarantee or a machine-cold test.

## Coverage by separately measured phase

| Phase | Comparator | Current medians faster | Minimum speedup | Maximum speedup |
| --- | --- | --- | --- | --- |
| First query | Default-thread brute | 18/18 | 23.808× | 162.091× |
| First query | Single-thread brute | 18/18 | 10.722× | 72.699× |
| First query | sklearn auto | 18/18 | 2.775× | 90.141× |
| First query | Preceding automatic | 18/18 | 7.460× | 43.585× |
| First fit + query | Default-thread brute | 18/18 | 19.786× | 65.161× |
| First fit + query | Single-thread brute | 18/18 | 9.161× | 30.245× |
| First fit + query | sklearn auto | 18/18 | 3.211× | 37.298× |
| First fit + query | Preceding automatic | 18/18 | 63.388× | 221.101× |
| Estimator import and setup | Default-thread brute | 0/18 | **0.983×** | **0.997×** |
| Estimator import and setup | Single-thread brute | 11/18 | **0.991×** | 1.017× |
| Estimator import and setup | sklearn auto | 0/18 | **0.981×** | **0.998×** |
| Estimator import and setup | Preceding automatic | 4/18 | **0.989×** | 1.004× |
| Estimator import through first result | Default-thread brute | 18/18 | 1.000× | 1.016× |
| Estimator import through first result | Single-thread brute | 17/18 | **0.998×** | 1.025× |
| Estimator import through first result | sklearn auto | 8/18 | **0.983×** | 1.013× |
| Estimator import through first result | Preceding automatic | 18/18 | 1.047× | 1.063× |
| Process start through first result | Default-thread brute | 16/18 | **0.998×** | 1.014× |
| Process start through first result | Single-thread brute | 17/18 | **0.999×** | 1.025× |
| Process start through first result | sklearn auto | 7/18 | **0.984×** | 1.012× |
| Process start through first result | Preceding automatic | 18/18 | 1.044× | 1.060× |

## First query

First public query after the newly constructed estimator's first fit.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.097 | 0.095–0.114 | 13.497 / 138.728× | 5.654 / 58.112× | 0.306 / 3.144× | 3.472 / 35.688× |
| Gaussian 32D, single query | 49 | 0.149 | 0.144–0.170 | 13.842 / 93.057× | 5.628 / 37.833× | 13.408 / 90.141× | 3.434 / 23.084× |
| Gaussian 8D, single query | 42 | 0.097 | 0.094–0.101 | 13.204 / 136.238× | 6.035 / 62.270× | 0.294 / 3.034× | 3.372 / 34.797× |
| Gaussian 32D, single query | 50 | 0.146 | 0.144–0.147 | 13.441 / 91.826× | 5.712 / 39.024× | 13.113 / 89.585× | 3.582 / 24.473× |
| Gaussian 8D, single query | 43 | 0.105 | 0.098–0.107 | 13.493 / 128.502× | 5.692 / 54.212× | 0.291 / 2.775× | 3.453 / 32.886× |
| Gaussian 32D, single query | 51 | 0.149 | 0.147–0.153 | 13.450 / 90.119× | 5.642 / 37.800× | 13.417 / 89.898× | 3.535 / 23.687× |
| Digits (raw) | 151 | 0.584 | 0.582–0.609 | 14.134 / 24.183× | 6.326 / 10.823× | 14.361 / 24.572× | 4.527 / 7.746× |
| Digits (raw) | 251 | 0.595 | 0.563–0.604 | 14.166 / 23.808× | 6.880 / 11.564× | 14.296 / 24.027× | 4.439 / 7.460× |
| Digits (raw) | 252 | 0.595 | 0.584–0.669 | 14.584 / 24.496× | 6.383 / 10.722× | 14.044 / 23.588× | 4.591 / 7.712× |
| Breast Cancer (raw) | 151 | 0.268 | 0.257–0.274 | 13.894 / 51.844× | 5.623 / 20.980× | 13.708 / 51.148× | 3.668 / 13.687× |
| Breast Cancer (raw) | 251 | 0.273 | 0.266–0.280 | 13.259 / 48.651× | 5.905 / 21.667× | 13.651 / 50.088× | 3.716 / 13.635× |
| Breast Cancer (raw) | 252 | 0.273 | 0.259–0.296 | 13.502 / 49.434× | 5.649 / 20.682× | 13.337 / 48.832× | 3.747 / 13.718× |
| Iris (raw) | 151 | 0.082 | 0.079–0.084 | 12.872 / 157.864× | 5.735 / 70.335× | 0.289 / 3.542× | 3.504 / 42.969× |
| Iris (raw) | 251 | 0.081 | 0.081–0.089 | 12.931 / 158.744× | 5.922 / 72.699× | 0.293 / 3.596× | 3.528 / 43.310× |
| Iris (raw) | 252 | 0.081 | 0.075–0.084 | 13.190 / 162.091× | 5.801 / 71.288× | 0.297 / 3.651× | 3.547 / 43.585× |
| Wine (raw) | 151 | 0.093 | 0.083–0.098 | 13.605 / 145.570× | 5.553 / 59.416× | 0.306 / 3.270× | 3.543 / 37.909× |
| Wine (raw) | 251 | 0.092 | 0.081–0.093 | 13.436 / 146.047× | 5.938 / 64.548× | 0.304 / 3.307× | 3.374 / 36.678× |
| Wine (raw) | 252 | 0.092 | 0.083–0.094 | 13.485 / 146.441× | 5.527 / 60.023× | 0.302 / 3.284× | 3.532 / 38.356× |

## First fit + query

Constructor, first fit and first query. Estimator imports/setup are separate.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.314 | 0.308–0.361 | 13.815 / 44.039× | 6.010 / 19.159× | 2.054 / 6.547× | 46.477 / 148.152× |
| Gaussian 32D, single query | 49 | 0.368 | 0.358–0.402 | 14.208 / 38.622× | 5.970 / 16.229× | 13.721 / 37.298× | 45.967 / 124.953× |
| Gaussian 8D, single query | 42 | 0.315 | 0.308–0.319 | 13.512 / 42.885× | 6.383 / 20.257× | 2.068 / 6.563× | 45.815 / 145.405× |
| Gaussian 32D, single query | 50 | 0.370 | 0.366–0.388 | 13.755 / 37.180× | 6.065 / 16.393× | 13.451 / 36.359× | 46.117 / 124.655× |
| Gaussian 8D, single query | 43 | 0.325 | 0.310–0.332 | 13.805 / 42.530× | 6.023 / 18.556× | 2.036 / 6.273× | 46.062 / 141.912× |
| Gaussian 32D, single query | 51 | 0.374 | 0.364–0.377 | 13.799 / 36.849× | 5.994 / 16.006× | 13.751 / 36.723× | 46.555 / 124.326× |
| Digits (raw) | 151 | 0.722 | 0.717–0.750 | 14.454 / 20.012× | 6.673 / 9.239× | 14.677 / 20.320× | 46.169 / 63.920× |
| Digits (raw) | 251 | 0.732 | 0.690–0.740 | 14.486 / 19.786× | 7.224 / 9.867× | 14.616 / 19.963× | 46.508 / 63.525× |
| Digits (raw) | 252 | 0.735 | 0.720–0.817 | 14.879 / 20.250× | 6.731 / 9.161× | 14.367 / 19.554× | 46.574 / 63.388× |
| Breast Cancer (raw) | 151 | 0.393 | 0.388–0.411 | 14.205 / 36.122× | 5.953 / 15.138× | 14.018 / 35.646× | 45.557 / 115.847× |
| Breast Cancer (raw) | 251 | 0.399 | 0.393–0.421 | 13.571 / 34.041× | 6.231 / 15.629× | 13.969 / 35.038× | 45.632 / 114.462× |
| Breast Cancer (raw) | 252 | 0.400 | 0.381–0.425 | 13.812 / 34.527× | 5.972 / 14.930× | 13.641 / 34.098× | 45.855 / 114.625× |
| Iris (raw) | 151 | 0.207 | 0.203–0.212 | 13.166 / 63.668× | 6.069 / 29.346× | 0.695 / 3.359× | 45.446 / 219.767× |
| Iris (raw) | 251 | 0.207 | 0.204–0.234 | 13.252 / 64.047× | 6.258 / 30.245× | 0.680 / 3.284× | 45.749 / 221.101× |
| Iris (raw) | 252 | 0.207 | 0.195–0.210 | 13.494 / 65.161× | 6.134 / 29.622× | 0.689 / 3.325× | 45.534 / 219.883× |
| Wine (raw) | 151 | 0.217 | 0.202–0.233 | 13.904 / 64.174× | 5.884 / 27.156× | 0.711 / 3.280× | 45.298 / 209.069× |
| Wine (raw) | 251 | 0.220 | 0.195–0.223 | 13.737 / 62.526× | 6.255 / 28.468× | 0.706 / 3.211× | 44.859 / 204.176× |
| Wine (raw) | 252 | 0.217 | 0.201–0.222 | 13.787 / 63.497× | 5.852 / 26.951× | 0.700 / 3.225× | 45.318 / 208.720× |

## Estimator import and setup

Normal estimator dependencies and configuration setup, including native initialization loaded by those imports.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 767.411 | 763.028–778.366 | 765.282 / **0.997×** | 780.074 / 1.017× | 765.742 / **0.998×** | 767.717 / 1.000× |
| Gaussian 32D, single query | 49 | 774.713 | 767.044–778.462 | 764.414 / **0.987×** | 773.701 / **0.999×** | 763.484 / **0.986×** | 776.060 / 1.002× |
| Gaussian 8D, single query | 42 | 775.952 | 768.040–778.417 | 765.607 / **0.987×** | 779.030 / 1.004× | 766.376 / **0.988×** | 768.752 / **0.991×** |
| Gaussian 32D, single query | 50 | 777.468 | 772.065–780.685 | 764.358 / **0.983×** | 770.690 / **0.991×** | 763.612 / **0.982×** | 778.178 / 1.001× |
| Gaussian 8D, single query | 43 | 772.782 | 769.753–797.932 | 765.055 / **0.990×** | 776.586 / 1.005× | 764.093 / **0.989×** | 769.528 / **0.996×** |
| Gaussian 32D, single query | 51 | 778.041 | 768.675–783.904 | 766.783 / **0.986×** | 778.216 / 1.000× | 767.146 / **0.986×** | 770.159 / **0.990×** |
| Digits (raw) | 151 | 775.011 | 769.134–795.801 | 768.527 / **0.992×** | 776.351 / 1.002× | 769.681 / **0.993×** | 774.706 / **1.000×** |
| Digits (raw) | 251 | 779.071 | 768.037–793.344 | 766.045 / **0.983×** | 786.507 / 1.010× | 769.885 / **0.988×** | 773.894 / **0.993×** |
| Digits (raw) | 252 | 773.286 | 767.043–819.836 | 768.127 / **0.993×** | 777.888 / 1.006× | 769.699 / **0.995×** | 776.536 / 1.004× |
| Breast Cancer (raw) | 151 | 776.466 | 768.687–795.581 | 767.456 / **0.988×** | 775.819 / **0.999×** | 764.315 / **0.984×** | 774.425 / **0.997×** |
| Breast Cancer (raw) | 251 | 775.486 | 771.926–784.865 | 764.371 / **0.986×** | 771.397 / **0.995×** | 769.868 / **0.993×** | 767.187 / **0.989×** |
| Breast Cancer (raw) | 252 | 780.592 | 771.698–789.080 | 770.209 / **0.987×** | 785.596 / 1.006× | 765.955 / **0.981×** | 774.769 / **0.993×** |
| Iris (raw) | 151 | 775.907 | 770.403–789.007 | 771.381 / **0.994×** | 779.899 / 1.005× | 773.060 / **0.996×** | 771.441 / **0.994×** |
| Iris (raw) | 251 | 781.836 | 774.792–793.670 | 769.310 / **0.984×** | 781.131 / **0.999×** | 768.035 / **0.982×** | 776.845 / **0.994×** |
| Iris (raw) | 252 | 772.959 | 772.017–792.023 | 765.752 / **0.991×** | 773.585 / 1.001× | 763.840 / **0.988×** | 770.066 / **0.996×** |
| Wine (raw) | 151 | 781.612 | 768.574–794.572 | 769.752 / **0.985×** | 776.383 / **0.993×** | 771.952 / **0.988×** | 775.824 / **0.993×** |
| Wine (raw) | 251 | 771.466 | 769.536–780.966 | 763.848 / **0.990×** | 778.271 / 1.009× | 761.484 / **0.987×** | 766.543 / **0.994×** |
| Wine (raw) | 252 | 773.341 | 772.726–779.782 | 763.159 / **0.987×** | 771.779 / **0.998×** | 768.471 / **0.994×** | 773.267 / **1.000×** |

## Estimator import through first result

Estimator import/setup plus first fit/query. Moving native initialization into imports does not remove it from this measure.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 767.724 | 763.338–778.728 | 779.655 / 1.016× | 786.668 / 1.025× | 767.809 / 1.000× | 814.007 / 1.060× |
| Gaussian 32D, single query | 49 | 775.115 | 767.412–778.820 | 778.622 / 1.005× | 779.932 / 1.006× | 777.205 / 1.003× | 822.027 / 1.061× |
| Gaussian 8D, single query | 42 | 776.263 | 768.348–778.736 | 779.119 / 1.004× | 785.090 / 1.011× | 768.441 / **0.990×** | 814.729 / 1.050× |
| Gaussian 32D, single query | 50 | 777.838 | 772.453–781.058 | 778.113 / 1.000× | 776.455 / **0.998×** | 777.291 / **0.999×** | 824.577 / 1.060× |
| Gaussian 8D, single query | 43 | 773.106 | 770.063–798.264 | 779.127 / 1.008× | 782.218 / 1.012× | 766.129 / **0.991×** | 815.685 / 1.055× |
| Gaussian 32D, single query | 51 | 778.405 | 769.041–784.280 | 780.079 / 1.002× | 784.021 / 1.007× | 781.489 / 1.004× | 817.041 / 1.050× |
| Digits (raw) | 151 | 775.734 | 769.851–796.521 | 783.269 / 1.010× | 782.721 / 1.009× | 784.695 / 1.012× | 820.875 / 1.058× |
| Digits (raw) | 251 | 779.803 | 768.777–794.034 | 780.530 / 1.001× | 793.742 / 1.018× | 784.194 / 1.006× | 819.717 / 1.051× |
| Digits (raw) | 252 | 774.006 | 767.778–820.654 | 783.420 / 1.012× | 784.619 / 1.014× | 783.847 / 1.013× | 823.110 / 1.063× |
| Breast Cancer (raw) | 151 | 776.864 | 769.076–795.992 | 781.500 / 1.006× | 781.669 / 1.006× | 778.333 / 1.002× | 819.982 / 1.056× |
| Breast Cancer (raw) | 251 | 775.880 | 772.331–785.286 | 778.020 / 1.003× | 777.059 / 1.002× | 783.725 / 1.010× | 812.032 / 1.047× |
| Breast Cancer (raw) | 252 | 781.017 | 772.080–789.477 | 783.524 / 1.003× | 791.568 / 1.014× | 779.475 / **0.998×** | 819.238 / 1.049× |
| Iris (raw) | 151 | 776.113 | 770.606–789.215 | 784.526 / 1.011× | 785.917 / 1.013× | 773.755 / **0.997×** | 816.573 / 1.052× |
| Iris (raw) | 251 | 782.053 | 774.997–793.905 | 782.276 / 1.000× | 787.390 / 1.007× | 768.692 / **0.983×** | 822.315 / 1.051× |
| Iris (raw) | 252 | 773.169 | 772.212–792.230 | 779.182 / 1.008× | 779.204 / 1.008× | 764.542 / **0.989×** | 815.214 / 1.054× |
| Wine (raw) | 151 | 781.814 | 768.791–794.806 | 783.079 / 1.002× | 782.267 / 1.001× | 772.657 / **0.988×** | 820.903 / 1.050× |
| Wine (raw) | 251 | 771.661 | 769.756–781.185 | 777.129 / 1.007× | 784.548 / 1.017× | 762.184 / **0.988×** | 811.840 / 1.052× |
| Wine (raw) | 252 | 773.563 | 772.928–780.002 | 777.377 / 1.005× | 777.644 / 1.005× | 769.176 / **0.994×** | 819.246 / 1.059× |

## Process start through first result

Parent-observed interpreter startup through the first result marker: includes NumPy import, fixture X/Q loading, estimator setup, fit/query and IPC; excludes the later audit and metadata.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Single-thread brute ms / ratio | sklearn auto ms / ratio | Preceding automatic ms / ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 818.556 | 813.349–832.632 | 829.637 / 1.014× | 838.653 / 1.025× | 818.277 / **1.000×** | 865.016 / 1.057× |
| Gaussian 32D, single query | 49 | 825.837 | 817.695–831.464 | 829.393 / 1.004× | 832.535 / 1.008× | 827.688 / 1.002× | 874.866 / 1.059× |
| Gaussian 8D, single query | 42 | 826.416 | 818.608–831.287 | 829.796 / 1.004× | 835.698 / 1.011× | 818.941 / **0.991×** | 865.361 / 1.047× |
| Gaussian 32D, single query | 50 | 829.299 | 822.885–833.196 | 829.495 / 1.000× | 828.266 / **0.999×** | 828.240 / **0.999×** | 877.017 / 1.058× |
| Gaussian 8D, single query | 43 | 823.339 | 821.578–849.530 | 830.306 / 1.008× | 833.478 / 1.012× | 816.574 / **0.992×** | 866.134 / 1.052× |
| Gaussian 32D, single query | 51 | 829.320 | 819.985–835.681 | 831.112 / 1.002× | 835.790 / 1.008× | 832.754 / 1.004× | 867.990 / 1.047× |
| Digits (raw) | 151 | 826.317 | 820.394–848.208 | 834.165 / 1.009× | 834.986 / 1.010× | 835.231 / 1.011× | 871.991 / 1.055× |
| Digits (raw) | 251 | 832.803 | 820.135–846.172 | 831.215 / **0.998×** | 845.852 / 1.016× | 835.405 / 1.003× | 871.342 / 1.046× |
| Digits (raw) | 252 | 825.404 | 818.562–871.803 | 833.942 / 1.010× | 836.415 / 1.013× | 834.986 / 1.012× | 874.666 / 1.060× |
| Breast Cancer (raw) | 151 | 827.629 | 820.106–846.921 | 833.338 / 1.007× | 834.516 / 1.008× | 828.381 / 1.001× | 871.378 / 1.053× |
| Breast Cancer (raw) | 251 | 827.265 | 822.515–836.305 | 829.173 / 1.002× | 828.232 / 1.001× | 834.275 / 1.008× | 863.472 / 1.044× |
| Breast Cancer (raw) | 252 | 831.628 | 822.208–842.499 | 834.111 / 1.003× | 843.994 / 1.015× | 830.022 / **0.998×** | 869.561 / 1.046× |
| Iris (raw) | 151 | 826.871 | 821.364–841.992 | 835.420 / 1.010× | 836.941 / 1.012× | 824.715 / **0.997×** | 867.947 / 1.050× |
| Iris (raw) | 251 | 832.980 | 825.974–845.749 | 832.957 / **1.000×** | 840.363 / 1.009× | 819.852 / **0.984×** | 872.876 / 1.048× |
| Iris (raw) | 252 | 824.152 | 823.873–842.828 | 829.810 / 1.007× | 830.359 / 1.008× | 815.507 / **0.990×** | 865.959 / 1.051× |
| Wine (raw) | 151 | 832.412 | 819.787–845.448 | 833.725 / 1.002× | 833.907 / 1.002× | 823.052 / **0.989×** | 871.631 / 1.047× |
| Wine (raw) | 251 | 822.649 | 820.315–834.070 | 827.600 / 1.006× | 836.246 / 1.017× | 812.789 / **0.988×** | 864.793 / 1.051× |
| Wine (raw) | 252 | 825.296 | 823.491–831.973 | 827.725 / 1.003× | 829.448 / 1.005× | 819.712 / **0.993×** | 869.658 / 1.054× |

## Every slower or missing median

Every below-1 or missing median from the five phase tables is repeated here. Paired ratios compare the corresponding serial launch rounds; their median need not equal the ratio of separate medians and is not a confidence interval.

| Phase | Comparator | Case | Current ms | Comparator ms | Median ratio | Paired ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Estimator import and setup | Default-thread brute | `shape_single_8_k5_seed41` | 767.411 | 765.282 | **0.997×** | **0.998×** |
| Estimator import and setup | Default-thread brute | `shape_single_32_k5_seed49` | 774.713 | 764.414 | **0.987×** | **0.986×** |
| Estimator import and setup | Default-thread brute | `shape_single_8_k5_seed42` | 775.952 | 765.607 | **0.987×** | **0.987×** |
| Estimator import and setup | Default-thread brute | `shape_single_32_k5_seed50` | 777.468 | 764.358 | **0.983×** | **0.982×** |
| Estimator import and setup | Default-thread brute | `shape_single_8_k5_seed43` | 772.782 | 765.055 | **0.990×** | **0.986×** |
| Estimator import and setup | Default-thread brute | `shape_single_32_k5_seed51` | 778.041 | 766.783 | **0.986×** | **0.987×** |
| Estimator import and setup | Default-thread brute | `real_digits_raw_k5_seed151` | 775.011 | 768.527 | **0.992×** | **0.991×** |
| Estimator import and setup | Default-thread brute | `real_digits_raw_k5_seed251` | 779.071 | 766.045 | **0.983×** | **0.982×** |
| Estimator import and setup | Default-thread brute | `real_digits_raw_k5_seed252` | 773.286 | 768.127 | **0.993×** | **0.992×** |
| Estimator import and setup | Default-thread brute | `real_breast_cancer_raw_k5_seed151` | 776.466 | 767.456 | **0.988×** | **0.985×** |
| Estimator import and setup | Default-thread brute | `real_breast_cancer_raw_k5_seed251` | 775.486 | 764.371 | **0.986×** | **0.983×** |
| Estimator import and setup | Default-thread brute | `real_breast_cancer_raw_k5_seed252` | 780.592 | 770.209 | **0.987×** | **0.987×** |
| Estimator import and setup | Default-thread brute | `real_iris_raw_k5_seed151` | 775.907 | 771.381 | **0.994×** | **0.986×** |
| Estimator import and setup | Default-thread brute | `real_iris_raw_k5_seed251` | 781.836 | 769.310 | **0.984×** | **0.978×** |
| Estimator import and setup | Default-thread brute | `real_iris_raw_k5_seed252` | 772.959 | 765.752 | **0.991×** | **0.991×** |
| Estimator import and setup | Default-thread brute | `real_wine_raw_k5_seed151` | 781.612 | 769.752 | **0.985×** | **0.991×** |
| Estimator import and setup | Default-thread brute | `real_wine_raw_k5_seed251` | 771.466 | 763.848 | **0.990×** | **0.992×** |
| Estimator import and setup | Default-thread brute | `real_wine_raw_k5_seed252` | 773.341 | 763.159 | **0.987×** | **0.986×** |
| Estimator import and setup | Single-thread brute | `shape_single_32_k5_seed49` | 774.713 | 773.701 | **0.999×** | 1.009× |
| Estimator import and setup | Single-thread brute | `shape_single_32_k5_seed50` | 777.468 | 770.690 | **0.991×** | **0.991×** |
| Estimator import and setup | Single-thread brute | `real_breast_cancer_raw_k5_seed151` | 776.466 | 775.819 | **0.999×** | **0.996×** |
| Estimator import and setup | Single-thread brute | `real_breast_cancer_raw_k5_seed251` | 775.486 | 771.397 | **0.995×** | **0.996×** |
| Estimator import and setup | Single-thread brute | `real_iris_raw_k5_seed251` | 781.836 | 781.131 | **0.999×** | 1.003× |
| Estimator import and setup | Single-thread brute | `real_wine_raw_k5_seed151` | 781.612 | 776.383 | **0.993×** | **0.997×** |
| Estimator import and setup | Single-thread brute | `real_wine_raw_k5_seed252` | 773.341 | 771.779 | **0.998×** | **0.999×** |
| Estimator import and setup | sklearn auto | `shape_single_8_k5_seed41` | 767.411 | 765.742 | **0.998×** | **0.997×** |
| Estimator import and setup | sklearn auto | `shape_single_32_k5_seed49` | 774.713 | 763.484 | **0.986×** | **0.991×** |
| Estimator import and setup | sklearn auto | `shape_single_8_k5_seed42` | 775.952 | 766.376 | **0.988×** | **0.988×** |
| Estimator import and setup | sklearn auto | `shape_single_32_k5_seed50` | 777.468 | 763.612 | **0.982×** | **0.985×** |
| Estimator import and setup | sklearn auto | `shape_single_8_k5_seed43` | 772.782 | 764.093 | **0.989×** | **0.989×** |
| Estimator import and setup | sklearn auto | `shape_single_32_k5_seed51` | 778.041 | 767.146 | **0.986×** | **0.984×** |
| Estimator import and setup | sklearn auto | `real_digits_raw_k5_seed151` | 775.011 | 769.681 | **0.993×** | **0.994×** |
| Estimator import and setup | sklearn auto | `real_digits_raw_k5_seed251` | 779.071 | 769.885 | **0.988×** | **0.993×** |
| Estimator import and setup | sklearn auto | `real_digits_raw_k5_seed252` | 773.286 | 769.699 | **0.995×** | **0.995×** |
| Estimator import and setup | sklearn auto | `real_breast_cancer_raw_k5_seed151` | 776.466 | 764.315 | **0.984×** | **0.984×** |
| Estimator import and setup | sklearn auto | `real_breast_cancer_raw_k5_seed251` | 775.486 | 769.868 | **0.993×** | **0.993×** |
| Estimator import and setup | sklearn auto | `real_breast_cancer_raw_k5_seed252` | 780.592 | 765.955 | **0.981×** | **0.984×** |
| Estimator import and setup | sklearn auto | `real_iris_raw_k5_seed151` | 775.907 | 773.060 | **0.996×** | **0.993×** |
| Estimator import and setup | sklearn auto | `real_iris_raw_k5_seed251` | 781.836 | 768.035 | **0.982×** | **0.986×** |
| Estimator import and setup | sklearn auto | `real_iris_raw_k5_seed252` | 772.959 | 763.840 | **0.988×** | **0.988×** |
| Estimator import and setup | sklearn auto | `real_wine_raw_k5_seed151` | 781.612 | 771.952 | **0.988×** | **0.991×** |
| Estimator import and setup | sklearn auto | `real_wine_raw_k5_seed251` | 771.466 | 761.484 | **0.987×** | **0.989×** |
| Estimator import and setup | sklearn auto | `real_wine_raw_k5_seed252` | 773.341 | 768.471 | **0.994×** | **0.990×** |
| Estimator import and setup | Preceding automatic | `shape_single_8_k5_seed42` | 775.952 | 768.752 | **0.991×** | **0.991×** |
| Estimator import and setup | Preceding automatic | `shape_single_8_k5_seed43` | 772.782 | 769.528 | **0.996×** | **0.991×** |
| Estimator import and setup | Preceding automatic | `shape_single_32_k5_seed51` | 778.041 | 770.159 | **0.990×** | **0.999×** |
| Estimator import and setup | Preceding automatic | `real_digits_raw_k5_seed151` | 775.011 | 774.706 | **1.000×** | **0.996×** |
| Estimator import and setup | Preceding automatic | `real_digits_raw_k5_seed251` | 779.071 | 773.894 | **0.993×** | **0.997×** |
| Estimator import and setup | Preceding automatic | `real_breast_cancer_raw_k5_seed151` | 776.466 | 774.425 | **0.997×** | **0.996×** |
| Estimator import and setup | Preceding automatic | `real_breast_cancer_raw_k5_seed251` | 775.486 | 767.187 | **0.989×** | **0.992×** |
| Estimator import and setup | Preceding automatic | `real_breast_cancer_raw_k5_seed252` | 780.592 | 774.769 | **0.993×** | **0.989×** |
| Estimator import and setup | Preceding automatic | `real_iris_raw_k5_seed151` | 775.907 | 771.441 | **0.994×** | 1.000× |
| Estimator import and setup | Preceding automatic | `real_iris_raw_k5_seed251` | 781.836 | 776.845 | **0.994×** | 1.001× |
| Estimator import and setup | Preceding automatic | `real_iris_raw_k5_seed252` | 772.959 | 770.066 | **0.996×** | **0.997×** |
| Estimator import and setup | Preceding automatic | `real_wine_raw_k5_seed151` | 781.612 | 775.824 | **0.993×** | **0.993×** |
| Estimator import and setup | Preceding automatic | `real_wine_raw_k5_seed251` | 771.466 | 766.543 | **0.994×** | **0.994×** |
| Estimator import and setup | Preceding automatic | `real_wine_raw_k5_seed252` | 773.341 | 773.267 | **1.000×** | **0.999×** |
| Estimator import through first result | Single-thread brute | `shape_single_32_k5_seed50` | 777.838 | 776.455 | **0.998×** | **0.999×** |
| Estimator import through first result | sklearn auto | `shape_single_8_k5_seed42` | 776.263 | 768.441 | **0.990×** | **0.990×** |
| Estimator import through first result | sklearn auto | `shape_single_32_k5_seed50` | 777.838 | 777.291 | **0.999×** | 1.001× |
| Estimator import through first result | sklearn auto | `shape_single_8_k5_seed43` | 773.106 | 766.129 | **0.991×** | **0.991×** |
| Estimator import through first result | sklearn auto | `real_breast_cancer_raw_k5_seed252` | 781.017 | 779.475 | **0.998×** | 1.001× |
| Estimator import through first result | sklearn auto | `real_iris_raw_k5_seed151` | 776.113 | 773.755 | **0.997×** | **0.994×** |
| Estimator import through first result | sklearn auto | `real_iris_raw_k5_seed251` | 782.053 | 768.692 | **0.983×** | **0.986×** |
| Estimator import through first result | sklearn auto | `real_iris_raw_k5_seed252` | 773.169 | 764.542 | **0.989×** | **0.989×** |
| Estimator import through first result | sklearn auto | `real_wine_raw_k5_seed151` | 781.814 | 772.657 | **0.988×** | **0.991×** |
| Estimator import through first result | sklearn auto | `real_wine_raw_k5_seed251` | 771.661 | 762.184 | **0.988×** | **0.990×** |
| Estimator import through first result | sklearn auto | `real_wine_raw_k5_seed252` | 773.563 | 769.176 | **0.994×** | **0.991×** |
| Process start through first result | Default-thread brute | `real_digits_raw_k5_seed251` | 832.803 | 831.215 | **0.998×** | **0.998×** |
| Process start through first result | Default-thread brute | `real_iris_raw_k5_seed251` | 832.980 | 832.957 | **1.000×** | **0.995×** |
| Process start through first result | Single-thread brute | `shape_single_32_k5_seed50` | 829.299 | 828.266 | **0.999×** | **0.999×** |
| Process start through first result | sklearn auto | `shape_single_8_k5_seed41` | 818.556 | 818.277 | **1.000×** | 1.000× |
| Process start through first result | sklearn auto | `shape_single_8_k5_seed42` | 826.416 | 818.941 | **0.991×** | **0.990×** |
| Process start through first result | sklearn auto | `shape_single_32_k5_seed50` | 829.299 | 828.240 | **0.999×** | 1.000× |
| Process start through first result | sklearn auto | `shape_single_8_k5_seed43` | 823.339 | 816.574 | **0.992×** | **0.992×** |
| Process start through first result | sklearn auto | `real_breast_cancer_raw_k5_seed252` | 831.628 | 830.022 | **0.998×** | 1.001× |
| Process start through first result | sklearn auto | `real_iris_raw_k5_seed151` | 826.871 | 824.715 | **0.997×** | **0.995×** |
| Process start through first result | sklearn auto | `real_iris_raw_k5_seed251` | 832.980 | 819.852 | **0.984×** | **0.989×** |
| Process start through first result | sklearn auto | `real_iris_raw_k5_seed252` | 824.152 | 815.507 | **0.990×** | **0.988×** |
| Process start through first result | sklearn auto | `real_wine_raw_k5_seed151` | 832.412 | 823.052 | **0.989×** | **0.991×** |
| Process start through first result | sklearn auto | `real_wine_raw_k5_seed251` | 822.649 | 812.789 | **0.988×** | **0.991×** |
| Process start through first result | sklearn auto | `real_wine_raw_k5_seed252` | 825.296 | 819.712 | **0.993×** | **0.990×** |

## Audits, actual routing and native initialization

| Configuration | Launches | Valid timings | Strict passes | Errors | Query backends | Native already loaded before fit |
| --- | --- | --- | --- | --- | --- | --- |
| Current automatic | 90 | 90 | 90 | 0 | {"cpu": 90} | {"True": 90} |
| Default-thread brute | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} |
| Single-thread brute | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} |
| sklearn auto | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} |
| Preceding automatic | 90 | 90 | 90 | 0 | {"cpu": 90} | {"False": 90} |

A true native-loaded-before-fit flag means normal estimator imports/setup loaded the extension. The setup and total-latency tables retain this cost. The harness performs no native prewarm. Each raw observation also records module presence before NumPy input preparation and after it.

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
| `shape_single_8_k5_seed41` | Current automatic | 24.741 | 1.688 | 0.216 | 796.027 | 893.734 |
| `shape_single_8_k5_seed41` | Default-thread brute | 24.741 | 1.639 | 0.335 | 807.614 | 912.931 |
| `shape_single_8_k5_seed41` | Single-thread brute | 25.219 | 1.655 | 0.338 | 815.544 | 916.105 |
| `shape_single_8_k5_seed41` | sklearn auto | 24.730 | 1.672 | 1.745 | 795.996 | 895.520 |
| `shape_single_8_k5_seed41` | Preceding automatic | 24.658 | 1.674 | 43.045 | 842.471 | 946.499 |
| `shape_single_32_k5_seed49` | Current automatic | 24.598 | 1.767 | 0.219 | 803.291 | 905.002 |
| `shape_single_32_k5_seed49` | Default-thread brute | 24.657 | 1.804 | 0.336 | 806.966 | 914.240 |
| `shape_single_32_k5_seed49` | Single-thread brute | 24.869 | 1.773 | 0.352 | 808.591 | 906.219 |
| `shape_single_32_k5_seed49` | sklearn auto | 24.678 | 1.800 | 0.330 | 805.620 | 911.411 |
| `shape_single_32_k5_seed49` | Preceding automatic | 24.679 | 1.777 | 42.529 | 851.224 | 955.006 |
| `shape_single_8_k5_seed42` | Current automatic | 24.627 | 1.657 | 0.217 | 804.364 | 902.329 |
| `shape_single_8_k5_seed42` | Default-thread brute | 24.687 | 1.657 | 0.315 | 807.552 | 912.827 |
| `shape_single_8_k5_seed42` | Single-thread brute | 24.698 | 1.669 | 0.348 | 813.290 | 911.719 |
| `shape_single_8_k5_seed42` | sklearn auto | 24.776 | 1.653 | 1.775 | 796.449 | 896.329 |
| `shape_single_8_k5_seed42` | Preceding automatic | 24.786 | 1.676 | 42.442 | 843.091 | 944.652 |
| `shape_single_32_k5_seed50` | Current automatic | 24.844 | 1.833 | 0.223 | 806.633 | 906.927 |
| `shape_single_32_k5_seed50` | Default-thread brute | 24.638 | 1.769 | 0.335 | 806.563 | 916.817 |
| `shape_single_32_k5_seed50` | Single-thread brute | 25.034 | 1.807 | 0.365 | 805.001 | 902.567 |
| `shape_single_32_k5_seed50` | sklearn auto | 24.753 | 1.844 | 0.341 | 805.574 | 914.313 |
| `shape_single_32_k5_seed50` | Preceding automatic | 25.534 | 1.840 | 42.645 | 853.464 | 957.854 |
| `shape_single_8_k5_seed43` | Current automatic | 24.812 | 1.690 | 0.218 | 801.204 | 901.925 |
| `shape_single_8_k5_seed43` | Default-thread brute | 24.856 | 1.715 | 0.314 | 807.731 | 913.739 |
| `shape_single_8_k5_seed43` | Single-thread brute | 24.630 | 1.702 | 0.328 | 810.416 | 907.239 |
| `shape_single_8_k5_seed43` | sklearn auto | 24.618 | 1.671 | 1.745 | 794.230 | 891.608 |
| `shape_single_8_k5_seed43` | Preceding automatic | 24.526 | 1.659 | 42.592 | 843.778 | 944.441 |
| `shape_single_32_k5_seed51` | Current automatic | 24.778 | 1.779 | 0.221 | 806.884 | 906.592 |
| `shape_single_32_k5_seed51` | Default-thread brute | 24.653 | 1.778 | 0.336 | 808.472 | 914.754 |
| `shape_single_32_k5_seed51` | Single-thread brute | 24.922 | 1.798 | 0.357 | 812.821 | 909.472 |
| `shape_single_32_k5_seed51` | sklearn auto | 24.981 | 1.783 | 0.333 | 810.227 | 915.869 |
| `shape_single_32_k5_seed51` | Preceding automatic | 24.843 | 1.799 | 42.982 | 845.308 | 952.714 |
| `real_digits_raw_k5_seed151` | Current automatic | 24.799 | 1.715 | 0.138 | 804.128 | 903.872 |
| `real_digits_raw_k5_seed151` | Default-thread brute | 24.698 | 1.748 | 0.311 | 811.319 | 917.053 |
| `real_digits_raw_k5_seed151` | Single-thread brute | 24.916 | 1.716 | 0.332 | 811.601 | 909.443 |
| `real_digits_raw_k5_seed151` | sklearn auto | 24.598 | 1.727 | 0.316 | 812.896 | 919.237 |
| `real_digits_raw_k5_seed151` | Preceding automatic | 24.961 | 1.734 | 41.509 | 848.999 | 951.454 |
| `real_digits_raw_k5_seed251` | Current automatic | 24.795 | 1.738 | 0.136 | 809.479 | 909.606 |
| `real_digits_raw_k5_seed251` | Default-thread brute | 24.747 | 1.683 | 0.318 | 808.657 | 917.528 |
| `real_digits_raw_k5_seed251` | Single-thread brute | 24.849 | 1.738 | 0.338 | 822.548 | 920.179 |
| `real_digits_raw_k5_seed251` | sklearn auto | 24.671 | 1.753 | 0.319 | 812.748 | 918.657 |
| `real_digits_raw_k5_seed251` | Preceding automatic | 25.050 | 1.731 | 42.059 | 848.307 | 952.070 |
| `real_digits_raw_k5_seed252` | Current automatic | 24.896 | 1.715 | 0.140 | 802.574 | 903.924 |
| `real_digits_raw_k5_seed252` | Default-thread brute | 24.732 | 1.705 | 0.321 | 811.462 | 918.394 |
| `real_digits_raw_k5_seed252` | Single-thread brute | 24.955 | 1.731 | 0.340 | 813.156 | 911.315 |
| `real_digits_raw_k5_seed252` | sklearn auto | 24.714 | 1.710 | 0.323 | 812.147 | 920.123 |
| `real_digits_raw_k5_seed252` | Preceding automatic | 24.924 | 1.680 | 41.987 | 851.653 | 954.867 |
| `real_breast_cancer_raw_k5_seed151` | Current automatic | 24.803 | 1.665 | 0.127 | 804.899 | 903.852 |
| `real_breast_cancer_raw_k5_seed151` | Default-thread brute | 24.666 | 1.667 | 0.304 | 810.711 | 917.276 |
| `real_breast_cancer_raw_k5_seed151` | Single-thread brute | 25.025 | 1.673 | 0.328 | 810.665 | 908.716 |
| `real_breast_cancer_raw_k5_seed151` | sklearn auto | 24.558 | 1.653 | 0.308 | 806.170 | 915.243 |
| `real_breast_cancer_raw_k5_seed151` | Preceding automatic | 24.972 | 1.632 | 41.886 | 848.161 | 951.610 |
| `real_breast_cancer_raw_k5_seed251` | Current automatic | 24.854 | 1.649 | 0.127 | 804.302 | 903.385 |
| `real_breast_cancer_raw_k5_seed251` | Default-thread brute | 24.805 | 1.627 | 0.306 | 806.365 | 914.539 |
| `real_breast_cancer_raw_k5_seed251` | Single-thread brute | 24.742 | 1.634 | 0.326 | 805.341 | 906.894 |
| `real_breast_cancer_raw_k5_seed251` | sklearn auto | 24.940 | 1.646 | 0.308 | 811.960 | 918.585 |
| `real_breast_cancer_raw_k5_seed251` | Preceding automatic | 24.942 | 1.678 | 42.069 | 840.669 | 945.278 |
| `real_breast_cancer_raw_k5_seed252` | Current automatic | 24.927 | 1.689 | 0.127 | 809.180 | 910.097 |
| `real_breast_cancer_raw_k5_seed252` | Default-thread brute | 24.673 | 1.666 | 0.311 | 811.757 | 919.204 |
| `real_breast_cancer_raw_k5_seed252` | Single-thread brute | 24.978 | 1.732 | 0.326 | 820.120 | 918.377 |
| `real_breast_cancer_raw_k5_seed252` | sklearn auto | 24.592 | 1.611 | 0.304 | 807.519 | 913.887 |
| `real_breast_cancer_raw_k5_seed252` | Preceding automatic | 24.887 | 1.636 | 42.108 | 847.230 | 949.807 |
| `real_iris_raw_k5_seed151` | Current automatic | 24.697 | 1.691 | 0.125 | 804.508 | 905.049 |
| `real_iris_raw_k5_seed151` | Default-thread brute | 24.726 | 1.598 | 0.299 | 812.872 | 919.281 |
| `real_iris_raw_k5_seed151` | Single-thread brute | 25.298 | 1.649 | 0.333 | 813.916 | 914.435 |
| `real_iris_raw_k5_seed151` | sklearn auto | 24.553 | 1.612 | 0.397 | 801.953 | 901.533 |
| `real_iris_raw_k5_seed151` | Preceding automatic | 24.836 | 1.594 | 41.987 | 845.297 | 949.221 |
| `real_iris_raw_k5_seed251` | Current automatic | 24.693 | 1.642 | 0.125 | 810.272 | 911.082 |
| `real_iris_raw_k5_seed251` | Default-thread brute | 24.632 | 1.646 | 0.307 | 810.531 | 916.556 |
| `real_iris_raw_k5_seed251` | Single-thread brute | 25.292 | 1.640 | 0.335 | 816.541 | 914.619 |
| `real_iris_raw_k5_seed251` | sklearn auto | 24.881 | 1.645 | 0.390 | 797.098 | 896.473 |
| `real_iris_raw_k5_seed251` | Preceding automatic | 24.803 | 1.722 | 42.221 | 850.257 | 954.324 |
| `real_iris_raw_k5_seed252` | Current automatic | 24.739 | 1.613 | 0.126 | 801.428 | 900.370 |
| `real_iris_raw_k5_seed252` | Default-thread brute | 24.543 | 1.604 | 0.300 | 807.463 | 913.792 |
| `real_iris_raw_k5_seed252` | Single-thread brute | 24.556 | 1.596 | 0.334 | 807.518 | 904.243 |
| `real_iris_raw_k5_seed252` | sklearn auto | 24.721 | 1.597 | 0.393 | 792.776 | 893.924 |
| `real_iris_raw_k5_seed252` | Preceding automatic | 24.681 | 1.624 | 42.109 | 843.434 | 945.697 |
| `real_wine_raw_k5_seed151` | Current automatic | 24.661 | 1.633 | 0.127 | 809.942 | 909.331 |
| `real_wine_raw_k5_seed151` | Default-thread brute | 24.656 | 1.655 | 0.300 | 811.258 | 917.240 |
| `real_wine_raw_k5_seed151` | Single-thread brute | 24.923 | 1.648 | 0.326 | 810.626 | 908.809 |
| `real_wine_raw_k5_seed151` | sklearn auto | 24.671 | 1.652 | 0.405 | 800.641 | 899.851 |
| `real_wine_raw_k5_seed151` | Preceding automatic | 24.604 | 1.587 | 41.755 | 849.340 | 950.754 |
| `real_wine_raw_k5_seed251` | Current automatic | 24.796 | 1.644 | 0.128 | 800.023 | 899.339 |
| `real_wine_raw_k5_seed251` | Default-thread brute | 24.642 | 1.633 | 0.304 | 805.256 | 914.006 |
| `real_wine_raw_k5_seed251` | Single-thread brute | 25.137 | 1.606 | 0.322 | 813.013 | 910.021 |
| `real_wine_raw_k5_seed251` | sklearn auto | 24.673 | 1.616 | 0.401 | 790.316 | 891.470 |
| `real_wine_raw_k5_seed251` | Preceding automatic | 24.952 | 1.643 | 41.516 | 840.959 | 945.483 |
| `real_wine_raw_k5_seed252` | Current automatic | 25.106 | 1.605 | 0.123 | 801.929 | 900.678 |
| `real_wine_raw_k5_seed252` | Default-thread brute | 24.734 | 1.638 | 0.302 | 805.184 | 912.466 |
| `real_wine_raw_k5_seed252` | Single-thread brute | 24.830 | 1.681 | 0.325 | 805.988 | 902.960 |
| `real_wine_raw_k5_seed252` | sklearn auto | 24.802 | 1.645 | 0.400 | 797.549 | 895.677 |
| `real_wine_raw_k5_seed252` | Preceding automatic | 24.611 | 1.635 | 41.786 | 847.394 | 951.144 |

## Evidence

[Every raw launch, audit and runtime identity](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-63db0ced93b3ab4b3e80)

[Validated derived summary](../benchmarks/results/knn-latency-final-2026-10-30-cold-summary.json)
