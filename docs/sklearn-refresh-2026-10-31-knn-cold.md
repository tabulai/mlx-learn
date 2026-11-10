# KNN process startup versus sklearn — October 31, 2026

The threadpool-limited worker requests `threadpool_limits(1)` and records one OpenMP thread. NumPy/SciPy use Apple Accelerate, which this threadpoolctl installation does not control; this is not a verified one-thread limit for every BLAS call. Default-thread workers retain the inherited environment.

Completed `2026-10-31T11:24:10.891289+00:00`. This report retains **360 separate process launches** over 18 fixtures, 4 configurations and 5 launches per fixture/configuration. Accuracy/integrity acceptance: **True**.

Each retained launch starts a new interpreter and performs exactly one first public NearestNeighbors fit/query. There are no warmups, metadata preflight requests, harness-driven native preimports or external pre-fit MLX synchronization. NumPy import and fixture X/Q loading are reported separately from estimator import/setup and API calls. Necessary transitive library imports remain charged. The first stdout result marker follows returned NumPy neighbors, before audits and runtime metadata. Parent process-start-to-result includes interpreter startup, fixture input loading, estimator imports, fit/query and result-marker IPC. Process elapsed also includes post-result audits/metadata/exit. This is process-cold, not a claim of cold OS/filesystem/Metal disk caches. All launches are serial.

Fit/query reductions may reflect moving native initialization into normal estimator imports. Estimator setup and estimator/process-to-result measures retain that cost. Parent result-marker latency includes interpreter/input loading/IPC; worker estimator-to-result excludes input preparation. No launch or failed audit is removed, and no OS/bytecode/Metal-disk-cache coldness is claimed.

All times are milliseconds. Speedup is comparator time / current automatic time; a value below 1× means current is slower. Bold ratios mark every slower or missing median. No failed numerical audit removes a timing from the comparison. These are descriptive process-cold measurements, not a universal first-use latency guarantee or a machine-cold test.

## Coverage by separately measured phase

| Phase | Comparator | Current medians faster | Minimum speedup | Maximum speedup |
| --- | --- | --- | --- | --- |
| First query | Default-thread brute | 18/18 | 20.835× | 142.911× |
| First query | Threadpool-limited brute | 18/18 | 9.361× | 56.147× |
| First query | sklearn auto | 18/18 | 3.201× | 82.281× |
| First fit + query | Default-thread brute | 18/18 | 9.086× | 13.583× |
| First fit + query | Threadpool-limited brute | 18/18 | 4.434× | 6.168× |
| First fit + query | sklearn auto | 18/18 | 1.485× | 11.789× |
| Estimator import and setup | Default-thread brute | 18/18 | 1.130× | 1.156× |
| Estimator import and setup | Threadpool-limited brute | 18/18 | 1.147× | 1.179× |
| Estimator import and setup | sklearn auto | 18/18 | 1.123× | 1.157× |
| Estimator import through first result | Default-thread brute | 18/18 | 1.161× | 1.189× |
| Estimator import through first result | Threadpool-limited brute | 18/18 | 1.159× | 1.192× |
| Estimator import through first result | sklearn auto | 18/18 | 1.124× | 1.188× |
| Process start through first result | Default-thread brute | 18/18 | 1.141× | 1.168× |
| Process start through first result | Threadpool-limited brute | 18/18 | 1.140× | 1.170× |
| Process start through first result | sklearn auto | 18/18 | 1.109× | 1.170× |

## First query

First public query after the newly constructed estimator's first fit.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Threadpool-limited brute ms / ratio | sklearn auto ms / ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.099 | 0.092–0.108 | 11.618 / 117.556× | 4.766 / 48.221× | 0.316 / 3.201× |
| Gaussian 32D, single query | 49 | 0.144 | 0.142–0.153 | 12.105 / 83.939× | 4.697 / 32.571× | 11.785 / 81.721× |
| Gaussian 8D, single query | 42 | 0.094 | 0.092–0.100 | 11.585 / 123.733× | 4.551 / 48.603× | 0.327 / 3.494× |
| Gaussian 32D, single query | 50 | 0.145 | 0.140–0.154 | 11.711 / 80.718× | 4.652 / 32.067× | 11.938 / 82.281× |
| Gaussian 8D, single query | 43 | 0.093 | 0.091–0.094 | 11.696 / 125.825× | 4.896 / 52.672× | 0.331 / 3.558× |
| Gaussian 32D, single query | 51 | 0.151 | 0.149–0.163 | 11.856 / 78.691× | 5.217 / 34.626× | 11.965 / 79.412× |
| Digits (raw) | 151 | 0.583 | 0.574–0.689 | 12.333 / 21.156× | 5.457 / 9.361× | 12.216 / 20.954× |
| Digits (raw) | 251 | 0.579 | 0.570–0.607 | 12.890 / 22.264× | 5.519 / 9.533× | 12.687 / 21.913× |
| Digits (raw) | 252 | 0.592 | 0.538–0.621 | 12.341 / 20.835× | 5.563 / 9.391× | 12.397 / 20.930× |
| Breast Cancer (raw) | 151 | 0.239 | 0.237–0.246 | 11.858 / 49.648× | 4.865 / 20.372× | 11.935 / 49.972× |
| Breast Cancer (raw) | 251 | 0.254 | 0.243–0.267 | 11.581 / 45.603× | 4.944 / 19.468× | 11.913 / 46.909× |
| Breast Cancer (raw) | 252 | 0.254 | 0.244–0.274 | 11.960 / 47.088× | 4.708 / 18.536× | 11.941 / 47.014× |
| Iris (raw) | 151 | 0.083 | 0.081–0.086 | 11.239 / 135.755× | 4.565 / 55.133× | 0.299 / 3.615× |
| Iris (raw) | 251 | 0.083 | 0.081–0.086 | 11.524 / 138.494× | 4.672 / 56.147× | 0.300 / 3.611× |
| Iris (raw) | 252 | 0.082 | 0.080–0.088 | 11.725 / 142.911× | 4.570 / 55.704× | 0.317 / 3.861× |
| Wine (raw) | 151 | 0.101 | 0.092–0.106 | 11.406 / 113.121× | 4.862 / 48.221× | 0.323 / 3.204× |
| Wine (raw) | 251 | 0.094 | 0.091–0.109 | 11.325 / 120.054× | 4.648 / 49.274× | 0.312 / 3.303× |
| Wine (raw) | 252 | 0.092 | 0.090–0.095 | 11.336 / 122.604× | 4.810 / 52.019× | 0.310 / 3.349× |

## First fit + query

Constructor, first fit and first query. Estimator imports/setup are separate.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Threadpool-limited brute ms / ratio | sklearn auto ms / ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 1.045 | 1.009–1.103 | 12.644 / 12.097× | 5.803 / 5.552× | 2.822 / 2.700× |
| Gaussian 32D, single query | 49 | 1.165 | 1.110–1.250 | 13.123 / 11.262× | 5.751 / 4.935× | 12.839 / 11.018× |
| Gaussian 8D, single query | 42 | 1.050 | 1.026–1.154 | 12.617 / 12.016× | 5.535 / 5.271× | 2.858 / 2.722× |
| Gaussian 32D, single query | 50 | 1.154 | 1.095–1.229 | 12.743 / 11.043× | 5.672 / 4.915× | 12.970 / 11.240× |
| Gaussian 8D, single query | 43 | 1.021 | 1.013–1.044 | 12.722 / 12.462× | 5.911 / 5.789× | 2.837 / 2.779× |
| Gaussian 32D, single query | 51 | 1.143 | 1.133–1.169 | 12.950 / 11.335× | 6.215 / 5.439× | 13.008 / 11.385× |
| Digits (raw) | 151 | 1.468 | 1.420–1.645 | 13.340 / 9.086× | 6.509 / 4.434× | 13.209 / 8.997× |
| Digits (raw) | 251 | 1.459 | 1.435–1.559 | 13.952 / 9.565× | 6.607 / 4.529× | 13.711 / 9.400× |
| Digits (raw) | 252 | 1.437 | 1.405–1.589 | 13.374 / 9.304× | 6.605 / 4.595× | 13.460 / 9.364× |
| Breast Cancer (raw) | 151 | 1.102 | 1.070–1.123 | 12.920 / 11.720× | 5.860 / 5.316× | 12.929 / 11.728× |
| Breast Cancer (raw) | 251 | 1.205 | 1.091–1.265 | 12.585 / 10.444× | 5.944 / 4.933× | 12.951 / 10.749× |
| Breast Cancer (raw) | 252 | 1.107 | 1.086–1.203 | 12.957 / 11.705× | 5.711 / 5.159× | 13.050 / 11.789× |
| Iris (raw) | 151 | 0.933 | 0.903–1.060 | 12.237 / 13.119× | 5.556 / 5.956× | 1.433 / 1.537× |
| Iris (raw) | 251 | 0.923 | 0.916–0.951 | 12.534 / 13.583× | 5.673 / 6.148× | 1.457 / 1.579× |
| Iris (raw) | 252 | 0.964 | 0.902–0.966 | 12.742 / 13.222× | 5.604 / 5.815× | 1.431 / 1.485× |
| Wine (raw) | 151 | 0.955 | 0.904–0.967 | 12.421 / 13.008× | 5.826 / 6.102× | 1.437 / 1.505× |
| Wine (raw) | 251 | 0.934 | 0.914–1.026 | 12.323 / 13.197× | 5.654 / 6.054× | 1.407 / 1.507× |
| Wine (raw) | 252 | 0.942 | 0.899–1.048 | 12.355 / 13.120× | 5.808 / 6.168× | 1.410 / 1.498× |

## Estimator import and setup

Normal estimator dependencies and configuration setup, including native initialization loaded by those imports.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Threadpool-limited brute ms / ratio | sklearn auto ms / ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 352.403 | 347.613–355.829 | 401.067 / 1.138× | 404.916 / 1.149× | 401.178 / 1.138× |
| Gaussian 32D, single query | 49 | 353.978 | 346.906–359.925 | 402.778 / 1.138× | 411.506 / 1.163× | 406.659 / 1.149× |
| Gaussian 8D, single query | 42 | 354.066 | 349.751–361.249 | 399.975 / 1.130× | 406.353 / 1.148× | 402.332 / 1.136× |
| Gaussian 32D, single query | 50 | 353.821 | 351.776–357.814 | 402.148 / 1.137× | 411.742 / 1.164× | 402.413 / 1.137× |
| Gaussian 8D, single query | 43 | 349.758 | 345.894–350.812 | 400.650 / 1.146× | 409.906 / 1.172× | 403.815 / 1.155× |
| Gaussian 32D, single query | 51 | 352.838 | 348.721–357.204 | 403.699 / 1.144× | 412.779 / 1.170× | 405.624 / 1.150× |
| Digits (raw) | 151 | 354.949 | 351.657–358.648 | 403.131 / 1.136× | 409.882 / 1.155× | 398.833 / 1.124× |
| Digits (raw) | 251 | 350.782 | 348.073–367.404 | 403.094 / 1.149× | 411.166 / 1.172× | 404.548 / 1.153× |
| Digits (raw) | 252 | 355.081 | 348.217–365.091 | 404.150 / 1.138× | 410.733 / 1.157× | 403.798 / 1.137× |
| Breast Cancer (raw) | 151 | 350.172 | 348.117–356.677 | 400.900 / 1.145× | 412.868 / 1.179× | 402.006 / 1.148× |
| Breast Cancer (raw) | 251 | 352.267 | 349.047–358.412 | 402.456 / 1.142× | 413.491 / 1.174× | 404.026 / 1.147× |
| Breast Cancer (raw) | 252 | 354.384 | 351.929–358.511 | 405.943 / 1.145× | 408.672 / 1.153× | 409.565 / 1.156× |
| Iris (raw) | 151 | 348.670 | 348.417–355.366 | 399.823 / 1.147× | 410.667 / 1.178× | 400.998 / 1.150× |
| Iris (raw) | 251 | 350.526 | 346.341–355.242 | 398.994 / 1.138× | 409.529 / 1.168× | 403.141 / 1.150× |
| Iris (raw) | 252 | 354.252 | 345.780–358.201 | 402.764 / 1.137× | 406.182 / 1.147× | 397.731 / 1.123× |
| Wine (raw) | 151 | 352.940 | 349.141–356.843 | 405.763 / 1.150× | 409.867 / 1.161× | 405.293 / 1.148× |
| Wine (raw) | 251 | 351.764 | 348.009–353.399 | 401.039 / 1.140× | 411.067 / 1.169× | 404.384 / 1.150× |
| Wine (raw) | 252 | 351.040 | 347.598–358.200 | 405.961 / 1.156× | 410.652 / 1.170× | 406.231 / 1.157× |

## Estimator import through first result

Estimator import/setup plus first fit/query. Moving native initialization into imports does not remove it from this measure.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Threadpool-limited brute ms / ratio | sklearn auto ms / ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 353.412 | 348.636–356.932 | 413.817 / 1.171× | 410.789 / 1.162× | 404.030 / 1.143× |
| Gaussian 32D, single query | 49 | 355.091 | 348.016–361.175 | 415.162 / 1.169× | 417.224 / 1.175× | 419.518 / 1.181× |
| Gaussian 8D, single query | 42 | 355.093 | 350.905–362.313 | 412.395 / 1.161× | 411.888 / 1.160× | 405.206 / 1.141× |
| Gaussian 32D, single query | 50 | 354.975 | 352.886–359.015 | 414.689 / 1.168× | 417.440 / 1.176× | 415.290 / 1.170× |
| Gaussian 8D, single query | 43 | 350.779 | 346.938–351.833 | 413.096 / 1.178× | 415.752 / 1.185× | 406.660 / 1.159× |
| Gaussian 32D, single query | 51 | 353.979 | 349.853–358.347 | 416.247 / 1.176× | 419.260 / 1.184× | 418.143 / 1.181× |
| Digits (raw) | 151 | 356.368 | 353.175–360.294 | 416.498 / 1.169× | 416.331 / 1.168× | 411.690 / 1.155× |
| Digits (raw) | 251 | 352.244 | 349.516–368.863 | 417.294 / 1.185× | 417.709 / 1.186× | 418.259 / 1.187× |
| Digits (raw) | 252 | 356.486 | 349.707–366.528 | 417.493 / 1.171× | 417.363 / 1.171× | 417.257 / 1.170× |
| Breast Cancer (raw) | 151 | 351.295 | 349.187–357.781 | 413.820 / 1.178× | 418.770 / 1.192× | 414.790 / 1.181× |
| Breast Cancer (raw) | 251 | 353.501 | 350.146–359.503 | 414.927 / 1.174× | 419.524 / 1.187× | 416.697 / 1.179× |
| Breast Cancer (raw) | 252 | 355.588 | 353.015–359.618 | 419.138 / 1.179× | 414.383 / 1.165× | 422.615 / 1.188× |
| Iris (raw) | 151 | 349.603 | 349.319–356.427 | 412.060 / 1.179× | 416.357 / 1.191× | 402.387 / 1.151× |
| Iris (raw) | 251 | 351.448 | 347.256–356.187 | 411.446 / 1.171× | 414.852 / 1.180× | 404.591 / 1.151× |
| Iris (raw) | 252 | 355.216 | 346.713–359.165 | 415.355 / 1.169× | 411.787 / 1.159× | 399.176 / 1.124× |
| Wine (raw) | 151 | 353.894 | 350.045–357.808 | 418.184 / 1.182× | 415.693 / 1.175× | 406.844 / 1.150× |
| Wine (raw) | 251 | 352.724 | 348.923–354.340 | 413.652 / 1.173× | 416.904 / 1.182× | 405.780 / 1.150× |
| Wine (raw) | 252 | 351.950 | 348.565–359.248 | 418.316 / 1.189× | 416.779 / 1.184× | 407.652 / 1.158× |

## Process start through first result

Parent-observed interpreter startup through the first result marker: includes NumPy import, fixture X/Q loading, estimator setup, fit/query and IPC; excludes the later audit and metadata.

Each comparator column shows its median milliseconds, then comparator/current speedup. The current range spans all retained process launches for that fixture.

| Fixture family | Seed | Current median ms | Current min–max ms | Default-thread brute ms / ratio | Threadpool-limited brute ms / ratio | sklearn auto ms / ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 399.054 | 393.413–403.172 | 460.092 / 1.153× | 456.268 / 1.143× | 448.720 / 1.124× |
| Gaussian 32D, single query | 49 | 402.669 | 393.163–408.181 | 461.992 / 1.147× | 463.708 / 1.152× | 466.614 / 1.159× |
| Gaussian 8D, single query | 42 | 400.981 | 395.601–409.280 | 457.562 / 1.141× | 457.171 / 1.140× | 451.334 / 1.126× |
| Gaussian 32D, single query | 50 | 402.737 | 398.868–405.428 | 459.965 / 1.142× | 462.299 / 1.148× | 461.149 / 1.145× |
| Gaussian 8D, single query | 43 | 395.675 | 392.009–398.262 | 457.966 / 1.157× | 461.831 / 1.167× | 451.512 / 1.141× |
| Gaussian 32D, single query | 51 | 400.535 | 396.155–404.503 | 462.711 / 1.155× | 464.080 / 1.159× | 465.277 / 1.162× |
| Digits (raw) | 151 | 402.427 | 398.386–405.689 | 461.422 / 1.147× | 461.381 / 1.146× | 456.848 / 1.135× |
| Digits (raw) | 251 | 398.039 | 394.297–416.883 | 463.099 / 1.163× | 464.023 / 1.166× | 463.012 / 1.163× |
| Digits (raw) | 252 | 403.595 | 394.966–412.895 | 462.950 / 1.147× | 463.998 / 1.150× | 462.560 / 1.146× |
| Breast Cancer (raw) | 151 | 396.588 | 394.237–404.009 | 459.387 / 1.158× | 463.968 / 1.170× | 461.110 / 1.163× |
| Breast Cancer (raw) | 251 | 398.737 | 395.854–405.723 | 459.687 / 1.153× | 464.672 / 1.165× | 462.689 / 1.160× |
| Breast Cancer (raw) | 252 | 401.513 | 398.891–406.614 | 464.302 / 1.156× | 460.192 / 1.146× | 469.708 / 1.170× |
| Iris (raw) | 151 | 395.154 | 394.193–402.463 | 457.297 / 1.157× | 461.812 / 1.169× | 447.565 / 1.133× |
| Iris (raw) | 251 | 396.995 | 392.401–402.063 | 456.696 / 1.150× | 460.728 / 1.161× | 451.254 / 1.137× |
| Iris (raw) | 252 | 399.957 | 392.144–405.710 | 461.416 / 1.154× | 457.426 / 1.144× | 443.523 / 1.109× |
| Wine (raw) | 151 | 400.214 | 396.072–404.806 | 463.190 / 1.157× | 460.386 / 1.150× | 452.440 / 1.130× |
| Wine (raw) | 251 | 398.305 | 394.727–400.492 | 458.691 / 1.152× | 462.792 / 1.162× | 451.874 / 1.134× |
| Wine (raw) | 252 | 398.090 | 393.699–405.862 | 465.161 / 1.168× | 461.929 / 1.160× | 452.859 / 1.138× |

## Every slower or missing median

Every below-1 or missing median from the five phase tables is repeated here. Paired ratios compare the corresponding serial launch rounds; their median need not equal the ratio of separate medians and is not a confidence interval.

No slower or missing medians in these five measured phases.

## Audits, actual routing and native initialization

| Configuration | Launches | Valid timings | Strict passes | Errors | Query backends | Native already loaded before fit | Stock sklearn.neighbors loaded before fit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current automatic | 90 | 90 | 90 | 0 | {"cpu": 90} | {"True": 90} | {"False": 90} |
| Default-thread brute | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |
| Threadpool-limited brute | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |
| sklearn auto | 90 | 90 | 65 | 0 | {"sklearn": 90} | {"False": 90} | {"True": 90} |

A true native-loaded-before-fit flag means normal estimator imports/setup loaded the extension. The setup and total-latency tables retain this cost. The harness performs no native prewarm. Each raw observation also records module presence before NumPy input preparation and after it. Stock sklearn.neighbors presence is recorded separately from the top-level sklearn package; older evidence lacking that field remains labelled not_recorded.

### Every failed observation

| Configuration | Case | Launch | Failure | Raw JSON pointer |
| --- | --- | --- | --- | --- |
| Default-thread brute | `real_digits_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/0` |
| Default-thread brute | `real_digits_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/1` |
| Default-thread brute | `real_digits_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/2` |
| Default-thread brute | `real_digits_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/3` |
| Default-thread brute | `real_digits_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_default/samples/4` |
| Threadpool-limited brute | `real_digits_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/0` |
| Threadpool-limited brute | `real_digits_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/1` |
| Threadpool-limited brute | `real_digits_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/2` |
| Threadpool-limited brute | `real_digits_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/3` |
| Threadpool-limited brute | `real_digits_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/6/variants/sklearn_1/samples/4` |
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
| Threadpool-limited brute | `real_digits_raw_k5_seed251` | 1 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/0` |
| Threadpool-limited brute | `real_digits_raw_k5_seed251` | 2 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/1` |
| Threadpool-limited brute | `real_digits_raw_k5_seed251` | 3 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/2` |
| Threadpool-limited brute | `real_digits_raw_k5_seed251` | 4 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/3` |
| Threadpool-limited brute | `real_digits_raw_k5_seed251` | 5 | valid_exact_tie_alternative | `/cases/7/variants/sklearn_1/samples/4` |
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
| Threadpool-limited brute | `real_digits_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/0` |
| Threadpool-limited brute | `real_digits_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/1` |
| Threadpool-limited brute | `real_digits_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/2` |
| Threadpool-limited brute | `real_digits_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/3` |
| Threadpool-limited brute | `real_digits_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/8/variants/sklearn_1/samples/4` |
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
| Threadpool-limited brute | `real_iris_raw_k5_seed151` | 1 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/0` |
| Threadpool-limited brute | `real_iris_raw_k5_seed151` | 2 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/1` |
| Threadpool-limited brute | `real_iris_raw_k5_seed151` | 3 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/2` |
| Threadpool-limited brute | `real_iris_raw_k5_seed151` | 4 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/3` |
| Threadpool-limited brute | `real_iris_raw_k5_seed151` | 5 | valid_exact_tie_alternative | `/cases/12/variants/sklearn_1/samples/4` |
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
| Threadpool-limited brute | `real_iris_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/0` |
| Threadpool-limited brute | `real_iris_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/1` |
| Threadpool-limited brute | `real_iris_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/2` |
| Threadpool-limited brute | `real_iris_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/3` |
| Threadpool-limited brute | `real_iris_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_1/samples/4` |
| sklearn auto | `real_iris_raw_k5_seed252` | 1 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/0` |
| sklearn auto | `real_iris_raw_k5_seed252` | 2 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/1` |
| sklearn auto | `real_iris_raw_k5_seed252` | 3 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/2` |
| sklearn auto | `real_iris_raw_k5_seed252` | 4 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/3` |
| sklearn auto | `real_iris_raw_k5_seed252` | 5 | valid_exact_tie_alternative | `/cases/14/variants/sklearn_auto/samples/4` |

## Input preparation, fit return and post-result process lifetime

These columns keep the remaining recorded phases visible. Fit return may precede deferred preparation charged to first query. Full process elapsed includes audits, metadata and process exit after the result marker, so it is not an inference-latency measure.

| Case | Configuration | NumPy import ms | Fixture load ms | First fit ms | Worker entry→result ms | Full process elapsed ms |
| --- | --- | --- | --- | --- | --- | --- |
| `shape_single_8_k5_seed41` | Current automatic | 18.724 | 1.275 | 0.953 | 375.215 | 454.856 |
| `shape_single_8_k5_seed41` | Default-thread brute | 18.446 | 1.288 | 1.029 | 436.024 | 524.668 |
| `shape_single_8_k5_seed41` | Threadpool-limited brute | 18.741 | 1.258 | 0.985 | 432.429 | 517.667 |
| `shape_single_8_k5_seed41` | sklearn auto | 18.346 | 1.258 | 2.502 | 425.386 | 511.557 |
| `shape_single_32_k5_seed49` | Current automatic | 18.903 | 1.434 | 1.021 | 377.956 | 460.837 |
| `shape_single_32_k5_seed49` | Default-thread brute | 19.014 | 1.451 | 1.019 | 437.396 | 526.854 |
| `shape_single_32_k5_seed49` | Threadpool-limited brute | 18.818 | 1.428 | 1.054 | 439.482 | 527.011 |
| `shape_single_32_k5_seed49` | sklearn auto | 19.068 | 1.406 | 1.058 | 442.036 | 531.392 |
| `shape_single_8_k5_seed42` | Current automatic | 18.961 | 1.257 | 0.950 | 377.028 | 457.814 |
| `shape_single_8_k5_seed42` | Default-thread brute | 18.473 | 1.278 | 1.019 | 433.704 | 522.125 |
| `shape_single_8_k5_seed42` | Threadpool-limited brute | 18.552 | 1.263 | 0.979 | 433.537 | 518.462 |
| `shape_single_8_k5_seed42` | sklearn auto | 18.735 | 1.263 | 2.515 | 426.949 | 513.418 |
| `shape_single_32_k5_seed50` | Current automatic | 18.977 | 1.455 | 1.011 | 377.842 | 459.476 |
| `shape_single_32_k5_seed50` | Default-thread brute | 18.743 | 1.386 | 1.060 | 436.388 | 524.254 |
| `shape_single_32_k5_seed50` | Threadpool-limited brute | 18.414 | 1.364 | 1.022 | 438.788 | 524.537 |
| `shape_single_32_k5_seed50` | sklearn auto | 18.389 | 1.417 | 1.038 | 436.862 | 530.340 |
| `shape_single_8_k5_seed43` | Current automatic | 18.680 | 1.271 | 0.927 | 372.094 | 452.031 |
| `shape_single_8_k5_seed43` | Default-thread brute | 18.485 | 1.281 | 1.025 | 434.260 | 522.681 |
| `shape_single_8_k5_seed43` | Threadpool-limited brute | 18.427 | 1.276 | 0.993 | 437.676 | 522.066 |
| `shape_single_8_k5_seed43` | sklearn auto | 18.356 | 1.264 | 2.486 | 427.892 | 513.803 |
| `shape_single_32_k5_seed51` | Current automatic | 18.785 | 1.426 | 0.994 | 375.951 | 457.985 |
| `shape_single_32_k5_seed51` | Default-thread brute | 18.707 | 1.381 | 1.057 | 438.236 | 527.830 |
| `shape_single_32_k5_seed51` | Threadpool-limited brute | 18.419 | 1.419 | 1.028 | 440.610 | 526.643 |
| `shape_single_32_k5_seed51` | sklearn auto | 19.276 | 1.435 | 1.049 | 440.804 | 532.705 |
| `real_digits_raw_k5_seed151` | Current automatic | 18.596 | 1.338 | 0.885 | 378.183 | 459.590 |
| `real_digits_raw_k5_seed151` | Default-thread brute | 18.874 | 1.316 | 1.008 | 437.943 | 526.084 |
| `real_digits_raw_k5_seed151` | Threadpool-limited brute | 18.566 | 1.279 | 1.010 | 437.827 | 522.421 |
| `real_digits_raw_k5_seed151` | sklearn auto | 18.426 | 1.278 | 1.004 | 433.157 | 523.529 |
| `real_digits_raw_k5_seed251` | Current automatic | 18.629 | 1.294 | 0.856 | 373.896 | 454.084 |
| `real_digits_raw_k5_seed251` | Default-thread brute | 18.526 | 1.294 | 1.034 | 439.241 | 530.077 |
| `real_digits_raw_k5_seed251` | Threadpool-limited brute | 18.731 | 1.371 | 1.017 | 439.531 | 526.371 |
| `real_digits_raw_k5_seed251` | sklearn auto | 18.509 | 1.268 | 1.049 | 439.482 | 531.557 |
| `real_digits_raw_k5_seed252` | Current automatic | 18.931 | 1.369 | 0.878 | 379.103 | 459.795 |
| `real_digits_raw_k5_seed252` | Default-thread brute | 18.398 | 1.313 | 1.033 | 438.861 | 529.595 |
| `real_digits_raw_k5_seed252` | Threadpool-limited brute | 19.226 | 1.368 | 1.026 | 439.476 | 530.551 |
| `real_digits_raw_k5_seed252` | sklearn auto | 19.079 | 1.319 | 1.035 | 438.755 | 527.769 |
| `real_breast_cancer_raw_k5_seed151` | Current automatic | 18.516 | 1.219 | 0.856 | 372.671 | 452.767 |
| `real_breast_cancer_raw_k5_seed151` | Default-thread brute | 18.583 | 1.211 | 1.014 | 435.672 | 525.968 |
| `real_breast_cancer_raw_k5_seed151` | Threadpool-limited brute | 18.491 | 1.239 | 0.989 | 440.166 | 523.678 |
| `real_breast_cancer_raw_k5_seed151` | sklearn auto | 18.649 | 1.205 | 1.029 | 437.006 | 524.866 |
| `real_breast_cancer_raw_k5_seed251` | Current automatic | 18.620 | 1.225 | 0.938 | 374.930 | 456.111 |
| `real_breast_cancer_raw_k5_seed251` | Default-thread brute | 18.338 | 1.188 | 1.003 | 436.105 | 524.332 |
| `real_breast_cancer_raw_k5_seed251` | Threadpool-limited brute | 18.746 | 1.217 | 1.003 | 440.803 | 525.350 |
| `real_breast_cancer_raw_k5_seed251` | sklearn auto | 19.226 | 1.266 | 1.037 | 438.561 | 529.853 |
| `real_breast_cancer_raw_k5_seed252` | Current automatic | 18.767 | 1.237 | 0.853 | 377.346 | 458.319 |
| `real_breast_cancer_raw_k5_seed252` | Default-thread brute | 19.024 | 1.245 | 1.000 | 440.537 | 530.875 |
| `real_breast_cancer_raw_k5_seed252` | Threadpool-limited brute | 18.523 | 1.239 | 1.003 | 435.911 | 519.961 |
| `real_breast_cancer_raw_k5_seed252` | sklearn auto | 18.849 | 1.246 | 1.027 | 445.601 | 533.920 |
| `real_iris_raw_k5_seed151` | Current automatic | 18.565 | 1.214 | 0.850 | 371.217 | 450.548 |
| `real_iris_raw_k5_seed151` | Default-thread brute | 18.555 | 1.198 | 0.989 | 433.366 | 521.971 |
| `real_iris_raw_k5_seed151` | Threadpool-limited brute | 18.582 | 1.230 | 1.013 | 437.988 | 521.527 |
| `real_iris_raw_k5_seed151` | sklearn auto | 18.646 | 1.208 | 1.123 | 423.684 | 509.582 |
| `real_iris_raw_k5_seed251` | Current automatic | 18.776 | 1.211 | 0.840 | 373.144 | 455.183 |
| `real_iris_raw_k5_seed251` | Default-thread brute | 18.769 | 1.212 | 1.010 | 433.130 | 521.523 |
| `real_iris_raw_k5_seed251` | Threadpool-limited brute | 18.389 | 1.208 | 0.979 | 437.015 | 522.459 |
| `real_iris_raw_k5_seed251` | sklearn auto | 18.595 | 1.197 | 1.137 | 427.521 | 514.250 |
| `real_iris_raw_k5_seed252` | Current automatic | 18.803 | 1.248 | 0.876 | 376.635 | 455.886 |
| `real_iris_raw_k5_seed252` | Default-thread brute | 18.785 | 1.220 | 1.034 | 437.192 | 526.853 |
| `real_iris_raw_k5_seed252` | Threadpool-limited brute | 18.845 | 1.222 | 1.026 | 433.498 | 517.236 |
| `real_iris_raw_k5_seed252` | sklearn auto | 18.350 | 1.194 | 1.113 | 420.250 | 505.391 |
| `real_wine_raw_k5_seed151` | Current automatic | 18.791 | 1.244 | 0.849 | 376.108 | 456.308 |
| `real_wine_raw_k5_seed151` | Default-thread brute | 18.347 | 1.206 | 1.011 | 439.436 | 528.298 |
| `real_wine_raw_k5_seed151` | Threadpool-limited brute | 18.638 | 1.222 | 0.979 | 436.898 | 521.503 |
| `real_wine_raw_k5_seed151` | sklearn auto | 18.711 | 1.204 | 1.117 | 428.677 | 515.698 |
| `real_wine_raw_k5_seed251` | Current automatic | 18.929 | 1.217 | 0.843 | 374.662 | 454.158 |
| `real_wine_raw_k5_seed251` | Default-thread brute | 18.497 | 1.199 | 0.998 | 435.113 | 523.489 |
| `real_wine_raw_k5_seed251` | Threadpool-limited brute | 18.736 | 1.225 | 0.987 | 438.972 | 523.401 |
| `real_wine_raw_k5_seed251` | sklearn auto | 18.952 | 1.200 | 1.096 | 427.685 | 513.805 |
| `real_wine_raw_k5_seed252` | Current automatic | 18.747 | 1.230 | 0.849 | 373.691 | 453.090 |
| `real_wine_raw_k5_seed252` | Default-thread brute | 18.715 | 1.218 | 1.014 | 440.620 | 529.545 |
| `real_wine_raw_k5_seed252` | Threadpool-limited brute | 18.667 | 1.225 | 0.999 | 438.151 | 525.269 |
| `real_wine_raw_k5_seed252` | sklearn auto | 18.415 | 1.219 | 1.096 | 428.970 | 513.485 |

## Evidence

[Every raw launch, audit and runtime identity](../benchmarks/results/sklearn-refresh-2026-10-31/knn-cold.json)

[Validated derived summary](../benchmarks/results/sklearn-refresh-2026-10-31/knn-cold-summary.json)
