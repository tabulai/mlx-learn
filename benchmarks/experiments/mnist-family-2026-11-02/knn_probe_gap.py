"""Drift probe: MNIST KNN predict back to back, after cache clears, gaps and rests.

Run alongside ``ioreg -r -d 1 -c IOAccelerator`` to see the GPU utilization other
processes leave behind; that load, not caching, moved predict between 0.11 and 0.22 s.
"""

import time

import mlx.core as mx
import numpy as np
from data import load

from mlxlearn.neighbors import KNeighborsClassifier

Xtr, ytr, Xte, _ = load("MNIST")
X, Q = (Xtr / 255.0).astype(np.float32), (Xte / 255.0).astype(np.float32)
m = KNeighborsClassifier(5).fit(X, ytr)
m.predict(Q)


def burst(tag, n=8, gap=0.0, clear=False):
    ts = []
    for _ in range(n):
        if clear:
            mx.clear_cache()
        time.sleep(gap)
        t0 = time.perf_counter()
        m.predict(Q)
        ts.append(time.perf_counter() - t0)
    print(f"{tag:22s}", " ".join(f"{t * 1e3:6.1f}" for t in ts),
          f"| cache={mx.get_cache_memory() / 2**20:.0f}MB active={mx.get_active_memory() / 2**20:.0f}MB")


time.sleep(5)
burst("back-to-back")
time.sleep(5)
burst("back-to-back+clear", clear=True)
time.sleep(5)
burst("gap 1s", gap=1.0)
time.sleep(10)
burst("after 10s rest")
burst("back-to-back x16", n=16)
