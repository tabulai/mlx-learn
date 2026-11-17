"""Order-effect probe: time MNIST KNN predict for input dtypes in the order given.

``python knn_probe_order.py 64 32`` then ``python knn_probe_order.py 32 64``. Whichever
variant runs second looked slower under background GPU load, regardless of dtype.
"""

import sys
import time

import numpy as np
from data import load

from mlxlearn.neighbors import KNeighborsClassifier

Xtr, ytr, Xte, _ = load("MNIST")
X64, Q64 = Xtr / 255.0, Xte / 255.0
data = {"32": (X64.astype(np.float32), Q64.astype(np.float32)), "64": (X64, Q64)}
KNeighborsClassifier(5).fit(data["32"][0], ytr).predict(data["32"][1])  # warm-up
for name in sys.argv[1:]:
    X, Q = data[name]
    m = KNeighborsClassifier(5).fit(X, ytr)
    ts = []
    for _ in range(6):
        t0 = time.perf_counter()
        m.predict(Q)
        ts.append(time.perf_counter() - t0)
    print(name, " ".join(f"{t * 1e3:6.1f}" for t in ts), "ms")
