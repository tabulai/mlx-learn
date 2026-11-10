"""mlx-learn vs scikit-learn KNeighborsClassifier on one MNIST-family dataset."""

import argparse
import time

import numpy as np
from data import SPLITS, load, write_json


def timed(fn, repeats):
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def run(label, make, Xtr, ytr, Xte, yte, k, fit_repeats, pred_repeats):
    fit_times, model = timed(lambda: make().fit(Xtr, ytr), fit_repeats)
    # First query on a fresh fit is reported separately: it pays any lazy device
    # upload of the training set and kernel compilation.
    t0 = time.perf_counter()
    model.predict(Xte)
    first = time.perf_counter() - t0
    pred_times, pred = timed(lambda: model.predict(Xte), pred_repeats)
    backend = getattr(model, "last_query_backend_", "sklearn")
    proba_times, proba = timed(lambda: model.predict_proba(Xte), pred_repeats)
    t0 = time.perf_counter()
    dist, idx = model.kneighbors(Xte, k)
    kn_time = time.perf_counter() - t0
    p_true = np.clip(proba[np.arange(len(yte)), np.searchsorted(model.classes_, yte)], 1e-15, 1)
    return {
        "label": label,
        "backend": backend,
        "fit_s": fit_times,
        "first_predict_s": first,
        "predict_s": pred_times,
        "predict_proba_s": proba_times,
        "kneighbors_s": kn_time,
        "test_acc": float((pred == yte).mean()),
        "test_logloss": float(-np.log(p_true).mean()),
        "pred": pred.tolist(),
        "idx": idx.tolist(),
        "dist": dist.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fit-repeats", type=int, default=5)
    ap.add_argument("--pred-repeats", type=int, default=3)
    ap.add_argument("--dataset", choices=sorted(SPLITS), default="MNIST")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load(args.dataset)
    # Same preprocessing as the logistic run: pixels scaled to [0, 1].
    Xtr64, Xte64 = Xtr / 255.0, Xte / 255.0
    Xtr32, Xte32 = Xtr64.astype(np.float32), Xte64.astype(np.float32)
    k, fr, pr = args.k, args.fit_repeats, args.pred_repeats

    import mlx.core as mx

    from mlxlearn.neighbors import KNeighborsClassifier as MLXKNN

    # Untimed warm-up so the first timed variant does not pay process-level MLX
    # initialization and kernel compilation.
    t0 = time.perf_counter()
    MLXKNN(n_neighbors=k).fit(Xtr32, ytr).predict(Xte32)
    warmup = time.perf_counter() - t0

    results = []
    r = run("mlx-learn (float32 input)", lambda: MLXKNN(n_neighbors=k), Xtr32, ytr, Xte32, yte, k, fr, pr)
    r["process_warmup_s"] = warmup
    results.append(r)
    results.append(run("mlx-learn (float64 input)", lambda: MLXKNN(n_neighbors=k), Xtr64, ytr, Xte64, yte, k, fr, pr))
    mx.clear_cache()
    time.sleep(2)

    from sklearn.neighbors import KNeighborsClassifier as SKKNN

    results.append(run("scikit-learn (float64 input)", lambda: SKKNN(n_neighbors=k), Xtr64, ytr, Xte64, yte, k, fr, pr))
    results.append(run("scikit-learn (float32 input)", lambda: SKKNN(n_neighbors=k), Xtr32, ytr, Xte32, yte, k, fr, pr))

    import sklearn

    meta = {
        "dataset": args.dataset,
        "k": k,
        "shapes": {"train": list(Xtr.shape), "test": list(Xte.shape)},
        "versions": {"sklearn": sklearn.__version__, "mlx": mx.__version__, "numpy": np.__version__},
    }
    write_json(args.out, {"meta": meta, "results": results})
    for r in results:
        print(f"{r['label']:30s} backend={r['backend']:7s} fit med={1e3*np.median(r['fit_s']):8.2f}ms  "
              f"first pred={r['first_predict_s']:7.3f}s  pred med={np.median(r['predict_s']):7.3f}s  "
              f"proba med={np.median(r['predict_proba_s']):7.3f}s  kneigh={r['kneighbors_s']:7.3f}s  "
              f"test={r['test_acc']:.4f}"
              + (f"  warmup={r['process_warmup_s']:.3f}s" if "process_warmup_s" in r else ""))


if __name__ == "__main__":
    main()
