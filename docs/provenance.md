# Source study and algorithm provenance

## SVC working-set optimization, October 31, 2026

The requested local `scikit-learn-intelex` checkout had most working-tree
sources marked deleted. Its committed snapshot was inspected through Git
objects without restoring or changing those files:
`f50ab692e6296b573efaaa3d05304a8eb3fbcf8d` (February 11, 2026).

The Python extension is a dispatch and compatibility layer around native
training. Its SVC selects `algorithm="thunder"`; its C++ binding forwards
kernel, precision, cache, tolerance and solver parameters into oneDAL. The
working-set implementation itself is in the external native backend, not the
Python wrapper. Sources: [Python SVM wrapper](https://github.com/tunguz/scikit-learn-intelex/blob/f50ab692e6296b573efaaa3d05304a8eb3fbcf8d/onedal/svm/svm.py),
[C++ SVM binding](https://github.com/tunguz/scikit-learn-intelex/blob/f50ab692e6296b573efaaa3d05304a8eb3fbcf8d/onedal/svm/svm.cpp).

The principal transferable idea is hierarchical decomposition: solve many SMO
steps inside a selected working set, then propagate their combined change to
the full gradient. This reduces repeated full-data scans and supports batched
kernel evaluation. The public mathematical description identifies the method
and its paper reference. [oneDAL SVM documentation](https://uxlfoundation.github.io/oneDAL/onedal/algorithms/svm/svm.html).

The native source review used oneDAL tag `2026.2.0`, commit
`480788c54a09ddd3dbc7cc9bc76d2ee4ea350020`. Its working-set selection retains
part of the preceding subset, the cache batches missing rows, and the training
loop alternates local optimization with global updates. Sources:
[working-set selection](https://github.com/uxlfoundation/oneDAL/blob/480788c54a09ddd3dbc7cc9bc76d2ee4ea350020/cpp/daal/src/algorithms/svm/svm_train_thunder_workset.h),
[row cache](https://github.com/uxlfoundation/oneDAL/blob/480788c54a09ddd3dbc7cc9bc76d2ee4ea350020/cpp/daal/src/algorithms/svm/svm_train_thunder_cache.h),
[training loop](https://github.com/uxlfoundation/oneDAL/blob/480788c54a09ddd3dbc7cc9bc76d2ee4ea350020/cpp/daal/src/algorithms/svm/svm_train_thunder_impl.i).

The algorithmic literature reference is Zeyi Wen, Jiashuai Shi, Qinbin Li,
Bingsheng He and Jian Chen, *ThunderSVM: A Fast SVM Library on GPUs and CPUs*,
JMLR 19(21):1–5, 2018. [Paper record](https://jmlr.org/papers/v19/17-740.html).

Persistent inference preparation is a second transferable pattern. The
reference wrapper retains its trained native model and reuses it on later
inference calls, invalidating it on coefficient/intercept assignment.
[Stored native model](https://github.com/tunguz/scikit-learn-intelex/blob/f50ab692e6296b573efaaa3d05304a8eb3fbcf8d/onedal/svm/svm.py#L129-L156),
[model invalidation](https://github.com/tunguz/scikit-learn-intelex/blob/f50ab692e6296b573efaaa3d05304a8eb3fbcf8d/sklearnex/svm/_base.py#L108-L132).

mlxlearn's implementations are independently written for its existing array,
precision and estimator contracts. No upstream source code was incorporated,
and no dependency on those libraries was introduced. In particular, mlxlearn
keeps float64 optimizer state and analytic diagonals, counts actual pair
updates, preserves the nonpositive-only curvature substitution, and requires
a reconstructed full KKT check before reporting convergence. It does not copy
wrapper shortcuts that substitute an iteration budget for measured iterations
or declare convergence unconditionally.

The reference project's published speedups are not benchmarks of this Apple
silicon implementation. Local before/after evidence and its limitations are
reported separately in [the working-set report](svc-working-set-2026-10-31.md).
