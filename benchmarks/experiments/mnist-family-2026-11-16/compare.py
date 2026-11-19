"""Markdown tables for the three-machine MNIST-family rerun.

Usage: ``python compare.py [label=results-dir ...]``. With no arguments it reads
``results/m1``, ``results/m4`` and ``results/m5`` and adds the committed 2026-11-02 M4
run (2026-11-10 for PCA) as ``ref``. Each directory holds the files ``run.sh`` and
``pca.py`` from ``../mnist-family-2026-11-02`` wrote.
"""

import gzip
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "mnist-family-2026-11-02"
DATASETS = [
    ("MNIST", "mnist"),
    ("Fashion-MNIST", "fashion_mnist"),
    ("Kannada-MNIST", "kannada_mnist"),
]
med = np.median

if len(sys.argv) > 1:
    RUNS = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}
else:
    RUNS = {
        "ref": BASE / "results",
        **{m.upper(): HERE / "results" / m for m in ("m1", "m4", "m5")},
    }
MACHINES = [m for m in RUNS if m != "ref"]


def read(path):
    if not path.exists():
        return None
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def get(machine, stem):
    return read(RUNS[machine] / f"{stem}.json.gz")


def pca_run(machine):
    d = RUNS[machine]
    for name in ("pca.json.gz", "pca.json", "pca-2026-11-10.json.gz"):
        if (d / name).exists():
            return read(d / name)
    return None


def lr_row(d):
    R = {r["label"]: r for r in d["results"]}
    return (
        R["mlx-learn (float32 input)"],
        R["mlx-learn (float64 input)"],
        R["scikit-learn (float64 input)"],
    )


def headline():
    """Speedups over scikit-learn, one cell per machine holding MNIST / Fashion / Kannada."""
    print("\n### Speedup over scikit-learn (MNIST / Fashion-MNIST / Kannada-MNIST)")
    print("| Benchmark | " + " | ".join(MACHINES) + " |")
    print("|---|" + "---|" * len(MACHINES))

    def cells(fn):
        out = []
        for m in MACHINES:
            vals = [fn(m, stem) for _, stem in DATASETS]
            out.append(
                " / ".join("-" if v is None else f"{v:.{0 if v >= 100 else 1}f}×" for v in vals)
            )
        return out

    def lr(max_iter):
        def f(m, stem):
            d = get(m, f"{stem}_logreg_iter{max_iter}")
            if not d:
                return None
            a, _, s = lr_row(d)
            return med(s["fit_s"]) / med(a["fit_s"])

        return f

    def knn(m, stem):
        d = get(m, f"{stem}_knn_k5")
        return (
            None if not d else min(d["results"][2]["predict_s"]) / min(d["results"][0]["predict_s"])
        )

    def svc(cache, op):
        def f(m, stem):
            a = get(m, f"{stem}_svc_mlx{cache}")
            s = get(m, f"{stem}_svc_sklearn{cache}")
            return None if not (a and s) else med(s[op]) / med(a[op])

        return f

    rows = [
        ("`LogisticRegression` fit, `max_iter=100`", lr(100)),
        ("`LogisticRegression` fit, `max_iter=5000`", lr(5000)),
        ("`KNeighborsClassifier` predict (best of 7)", knn),
        ("`SVC` fit, 200 MB cache", svc("", "fit_s")),
        ("`SVC` fit, 2,000 MB cache", svc("_c2000", "fit_s")),
        ("`SVC` predict", svc("", "predict_s")),
    ]
    for label, fn in rows:
        print(f"| {label} | " + " | ".join(cells(fn)) + " |")

    def pca_range(solver, dtype, op, digits=1):
        out = []
        for m in MACHINES:
            ev = pca_run(m)
            v = [
                c["sklearn"]["median_ms"][op] / c["mlxlearn"]["median_ms"][op]
                for c in ev["cases"]
                if c["solver"] == solver
                and c["dtype"] == dtype
                and (op != "transform_test" or c["dataset"] != "Kannada_MNIST")
            ]
            out.append(f"{min(v):.{digits}f}–{max(v):.{digits}f}×")
        return out

    for solver, label in (("auto", "default solver"), ("randomized", "`randomized`")):
        for dtype in ("float32", "float64"):
            print(
                f"| `PCA` fit, {label}, {dtype} | "
                + " | ".join(pca_range(solver, dtype, "fit_train"))
                + " |"
            )
    for dtype in ("float32", "float64"):
        print(
            f"| `PCA` 10,000-row test transform, {dtype} | "
            + " | ".join(pca_range("auto", dtype, "transform_test", 2))
            + " |"
        )


def absolute():
    print("\n### mlxlearn times (MNIST / Fashion-MNIST / Kannada-MNIST)")
    print("| | " + " | ".join(MACHINES) + " |")
    print("|---|" + "---|" * len(MACHINES))

    def row(label, fn, fmt):
        cells = []
        for m in MACHINES:
            cells.append(" / ".join(fmt(fn(m, stem)) for _, stem in DATASETS))
        print(f"| {label} | " + " | ".join(cells) + " |")

    row(
        "`LogisticRegression` fit, `max_iter=5000`",
        lambda m, s: med(lr_row(get(m, f"{s}_logreg_iter5000"))[0]["fit_s"]),
        lambda v: f"{v:.2f} s",
    )
    row(
        "`KNeighborsClassifier` predict (best)",
        lambda m, s: 1e3 * min(get(m, f"{s}_knn_k5")["results"][0]["predict_s"]),
        lambda v: f"{v:.0f} ms",
    )
    row(
        "`SVC` fit, 200 MB",
        lambda m, s: med(get(m, f"{s}_svc_mlx")["fit_s"]),
        lambda v: f"{v:.1f} s",
    )
    row(
        "`SVC` fit, 2,000 MB",
        lambda m, s: med(get(m, f"{s}_svc_mlx_c2000")["fit_s"]),
        lambda v: f"{v:.1f} s",
    )
    row(
        "`SVC` predict",
        lambda m, s: med(get(m, f"{s}_svc_mlx")["predict_s"]),
        lambda v: f"{v:.2f} s",
    )
    row(
        "scikit-learn `KNeighborsClassifier` predict (float64, best)",
        lambda m, s: min(get(m, f"{s}_knn_k5")["results"][2]["predict_s"]),
        lambda v: f"{v:.2f} s",
    )
    row(
        "scikit-learn `SVC` fit, 200 MB",
        lambda m, s: med(get(m, f"{s}_svc_sklearn")["fit_s"]),
        lambda v: f"{v:.0f} s",
    )


def lr_detail(max_iter):
    print(f"\n### LogisticRegression, max_iter={max_iter} (median of 5 fits)")
    print(
        "| dataset | machine | mlx f32 fit | mlx f64 fit | sk fit | iters mlx / sk | conv mlx / sk "
        "| test acc mlx / sk | obj (mlx-sk)/sk | pred agree |"
    )
    print("|" + "---|" * 10)
    for name, stem in DATASETS:
        for m in RUNS:
            d = get(m, f"{stem}_logreg_iter{max_iter}")
            if not d:
                continue
            a, b, s = lr_row(d)
            agree = np.mean(np.array(a["pred"]) == np.array(s["pred"]))
            print(
                f"| {name} | {m} | {med(a['fit_s']):.3f} s | {med(b['fit_s']):.3f} s | {med(s['fit_s']):.3f} s "
                f"| {a['n_iter']} / {s['n_iter']} | {a['converged']} / {s['converged']} "
                f"| {a['test_acc']:.2%} / {s['test_acc']:.2%} "
                f"| {(a['objective'] - s['objective']) / s['objective']:+.1e} | {agree:.2%} |"
            )


def knn_detail():
    print("\n### KNeighborsClassifier k=5, predict best / median of 7")
    print(
        "| dataset | machine | mlx f32 | mlx f64 | sk f64 | sk f32 | first mlx predict | test acc | pred agree |"
    )
    print("|" + "---|" * 9)
    for name, stem in DATASETS:
        for m in RUNS:
            d = get(m, f"{stem}_knn_k5")
            if not d:
                continue
            R = d["results"]
            p = [1e3 * np.array(r["predict_s"]) for r in R]
            accs = sorted({round(r["test_acc"], 6) for r in R})
            agree = min(np.mean(np.array(r["pred"]) == np.array(R[2]["pred"])) for r in R)
            print(
                f"| {name} | {m} | "
                + " | ".join(f"{x.min():.0f} / {med(x):.0f} ms" for x in p)
                + f" | {1e3 * R[0]['first_predict_s']:.0f} ms | {', '.join(f'{x:.2%}' for x in accs)} | {agree:.2%} |"
            )


def svc_detail():
    print("\n### SVC (rbf, C=1, gamma=scale); mlx median of 3, sklearn single run")
    print(
        "| dataset | machine | mlx fit 200 / 2,000 MB | sk fit 200 / 2,000 MB | mlx predict | sk predict "
        "| test acc mlx / sk | nSV mlx / sk | pred agree |"
    )
    print("|" + "---|" * 9)
    for name, stem in DATASETS:
        for m in RUNS:
            a, ac, s, sc = (
                get(m, f"{stem}_svc_{v}") for v in ("mlx", "mlx_c2000", "sklearn", "sklearn_c2000")
            )
            if not (a and ac and s and sc):
                continue
            agree = np.mean(np.array(a["pred"]) == np.array(s["pred"]))
            print(
                f"| {name} | {m} | {med(a['fit_s']):.2f} / {med(ac['fit_s']):.2f} s "
                f"| {med(s['fit_s']):.1f} / {med(sc['fit_s']):.1f} s | {med(a['predict_s']):.3f} s "
                f"| {med(s['predict_s']):.2f} s | {a['test_acc']:.2%} / {s['test_acc']:.2%} "
                f"| {sum(a['n_support'])} / {sum(s['n_support'])} | {agree:.2%} |"
            )


def pca_detail():
    runs = {m: pca_run(m) for m in RUNS}
    runs = {m: ev for m, ev in runs.items() if ev}
    print(
        "\n### PCA medians in ms, sklearn / mlxlearn (speedup), and the largest principal angle "
        "against float64 scikit-learn, mlxlearn / sklearn"
    )
    print(
        "| dataset | dtype | solver | k | "
        + " | ".join(f"{m} fit" for m in runs)
        + " | "
        + " | ".join(f"{m} test transform" for m in runs)
        + " | "
        + " | ".join(f"{m} angle °" for m in runs)
        + " |"
    )
    print("|" + "---|" * (4 + 3 * len(runs)))
    first = next(iter(runs.values()))
    for c0 in first["cases"]:
        key = (c0["dataset"], c0["dtype"], c0["solver"], c0["components"])
        fits, tfs, angles = [], [], []
        for ev in runs.values():
            c = next(
                c
                for c in ev["cases"]
                if (c["dataset"], c["dtype"], c["solver"], c["components"]) == key
            )
            s, x = c["sklearn"], c["mlxlearn"]
            fs, fm = s["median_ms"]["fit_train"], x["median_ms"]["fit_train"]
            ts, tm = s["median_ms"]["transform_test"], x["median_ms"]["transform_test"]
            fits.append(f"{fs:.0f} / {fm:.1f} ({fs / fm:.1f}×)")
            tfs.append(f"{ts:.2f} / {tm:.2f} ({ts / tm:.2f}×)")
            angles.append(
                f"{x['quality']['max_principal_angle_deg']:.2g} / {s['quality']['max_principal_angle_deg']:.2g}"
            )
        name = key[0].replace("_", "-")
        print(
            f"| {name} | {key[1]} | {key[2]} | {key[3]} | " + " | ".join(fits + tfs + angles) + " |"
        )
    if "M5" in runs:
        print("\nM5 PCA fit against the pre-binary32 baseline in the same run (`before`, 1859e09):")
        for c in runs["M5"]["cases"]:
            if c["dataset"] != "MNIST":
                continue
            a, b = c["mlxlearn"], c["before"]
            ratio = statistics.median(a["ms"]["fit_train"]) / statistics.median(
                b["ms"]["fit_train"]
            )
            print(
                f"- {c['dtype']} {c['solver']} k={c['components']}: fit {ratio:.2f}× the baseline time, angle "
                f"{a['quality']['max_principal_angle_deg']:.2g}° vs {b['quality']['max_principal_angle_deg']:.2g}°"
            )


if __name__ == "__main__":
    headline()
    absolute()
    lr_detail(100)
    lr_detail(5000)
    knn_detail()
    svc_detail()
    pca_detail()
