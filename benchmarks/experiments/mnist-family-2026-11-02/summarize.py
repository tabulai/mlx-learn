"""Markdown summary of the logreg, KNN and SVC results across the three datasets.

Usage: ``python summarize.py [results-dir]`` (default: the committed ``results/``).
"""

import sys
from pathlib import Path

import numpy as np
from data import RESULTS, SPLITS, read_json

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
med = np.median


def get(ds, run):
    p = DIR / f"{ds.lower()}_{run}.json.gz"
    return read_json(p) if p.exists() else None


def lr_table(max_iter):
    print(f"\n### Logistic regression, max_iter={max_iter}")
    print("| dataset | mlx fit | sk fit | fit speedup | mlx iters | sk iters | mlx conv | sk conv | mlx test | sk test | mlx obj | sk obj | pred agree | mlx predict (f64) | sk predict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in SPLITS:
        d = get(ds, f"logreg_iter{max_iter}")
        if not d:
            continue
        R = {r["label"]: r for r in d["results"]}
        m, m64, s = R["mlx-learn (float32 input)"], R["mlx-learn (float64 input)"], R["scikit-learn (float64 input)"]
        agree = np.mean(np.array(m["pred"]) == np.array(s["pred"]))
        print(f"| {ds} | {med(m['fit_s']):.3f} s | {med(s['fit_s']):.3f} s | {med(s['fit_s'])/med(m['fit_s']):.1f}x "
              f"| {m['n_iter']} | {s['n_iter']} | {m['converged']} | {s['converged']} "
              f"| {m['test_acc']:.2%} | {s['test_acc']:.2%} | {m['objective']:.6f} | {s['objective']:.6f} | {agree:.2%} "
              f"| {1e3*med(m64['predict_s']):.1f} ms | {1e3*med(s['predict_s']):.1f} ms |")


def knn_table():
    print("\n### KNN (k=5)")
    print("| dataset | mlx f32 predict best/med | mlx f64 predict best/med | sk f64 predict best/med | sk f32 predict med | speedup (best) | test acc (all) | pred agree |")
    print("|---|---|---|---|---|---|---|---|")
    for ds in SPLITS:
        d = get(ds, "knn_k5")
        if not d:
            continue
        R = d["results"]
        p = [np.array(r["predict_s"]) for r in R]
        accs = sorted({round(r["test_acc"], 6) for r in R})
        agree = min(np.mean(np.array(r["pred"]) == np.array(R[2]["pred"])) for r in R)
        print(f"| {ds} | {p[0].min():.3f}/{med(p[0]):.3f} s | {p[1].min():.3f}/{med(p[1]):.3f} s "
              f"| {p[2].min():.3f}/{med(p[2]):.3f} s | {med(p[3]):.3f} s | {p[2].min()/p[0].min():.1f}x "
              f"| {', '.join(f'{a:.2%}' for a in accs)} | {agree:.2%} |")


def svc_table():
    print("\n### SVC (rbf, C=1, gamma=scale)")
    print("| dataset | mlx fit (200 MB) | mlx fit (2000 MB) | sk fit (200 / 2000 MB) | fit speedup (200 / 2000 MB) | mlx predict | sk predict | predict speedup | mlx test | sk test | nSV mlx/sk | pred agree |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in SPLITS:
        m, mc, s, sc = (get(ds, f"svc_{v}") for v in ("mlx", "mlx_c2000", "sklearn", "sklearn_c2000"))
        if not (m and s):
            continue
        # The 2,000 MB speedup compares both libraries at that cache size.
        sk_fit2 = med(sc["fit_s"]) if sc else float("nan")
        sk2 = f"{sk_fit2:.1f}" if sc else "-"
        mc_fit = med(mc["fit_s"]) if mc else float("nan")
        agree = np.mean(np.array(m["pred"]) == np.array(s["pred"]))
        print(f"| {ds} | {med(m['fit_s']):.2f} s | {mc_fit:.2f} s | {med(s['fit_s']):.1f} / {sk2} s "
              f"| {med(s['fit_s'])/med(m['fit_s']):.1f}x / {sk_fit2/mc_fit:.1f}x "
              f"| {med(m['predict_s']):.3f} s | {med(s['predict_s']):.2f} s | {med(s['predict_s'])/med(m['predict_s']):.0f}x "
              f"| {m['test_acc']:.2%} | {s['test_acc']:.2%} | {sum(m['n_support'])}/{sum(s['n_support'])} | {agree:.2%} |")


lr_table(100)
lr_table(5000)
knn_table()
svc_table()
