"""Exact neighbor audit of a ``knn.py`` result file.

Raw pixels are integers, so float64 GEMM on them gives exact integer squared distances
(every partial sum is below 2**53). Scaling by 1/255 preserves the neighbor order.
"""

import sys

import numpy as np
from data import load, read_json

res = read_json(sys.argv[1])
Xtr, _, Xte, _ = load(res["meta"].get("dataset", "MNIST"))  # float64 holding integers 0..255
k = res["meta"]["k"]
nte = len(Xte)
xx = (Xtr**2).sum(1)
ref_kth = np.empty((nte, k))
for s in range(0, nte, 1000):
    Q = Xte[s:s + 1000]
    D = (Q**2).sum(1)[:, None] + xx[None, :] - 2.0 * (Q @ Xtr.T)
    assert np.all(D == np.round(D)) and D.min() >= 0
    ref_kth[s:s + 1000] = np.sort(np.partition(D, k - 1, axis=1)[:, :k], axis=1)
sk_idx = np.array(res["results"][2]["idx"])  # scikit-learn, float64 input
for r in res["results"]:
    idx = np.array(r["idx"])
    exact = ((Xte[:, None, :] - Xtr[idx]) ** 2).sum(-1)  # exact integer distances of returned rows
    got = np.sort(exact, axis=1)
    correct_sets = np.all(got == ref_kth, axis=1).mean()
    same_idx_as_sk64 = np.all(np.sort(idx, 1) == np.sort(sk_idx, 1), axis=1).mean()
    ordered = np.all(np.diff(exact, axis=1) >= 0, axis=1).mean()
    dist = np.array(r["dist"]) * 255.0
    derr = np.abs(dist - np.sqrt(exact)).max() / 255.0
    print(f"{r['label']:30s} exact-neighbor rows={correct_sets:.4%}  ordered={ordered:.4%}  "
          f"same set as sklearn64={same_idx_as_sk64:.4%}  max|dist err|={derr:.2e}")
