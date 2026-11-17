"""mlx-learn vs scikit-learn LogisticRegression on one MNIST-family dataset."""

import argparse
import time
import warnings

import numpy as np
from data import SPLITS, load, write_json


def timed(fn, repeats):
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def objective(model, X, y):
    """scikit-learn's multinomial lbfgs objective and max|gradient|, in float64."""
    X = np.asarray(X, dtype=np.float64)
    W, b, n = model.coef_, model.intercept_, X.shape[0]
    Z = X @ W.T + b
    Z -= Z.max(axis=1, keepdims=True)
    logP = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    yi = np.searchsorted(model.classes_, y)
    reg = 1.0 / (model.C * n)
    obj = -logP[np.arange(n), yi].mean() + 0.5 * reg * (W ** 2).sum()
    R = np.exp(logP)
    R[np.arange(n), yi] -= 1.0
    gW = R.T @ X / n + reg * W
    gb = R.mean(axis=0)
    return float(obj), float(max(np.abs(gW).max(), np.abs(gb).max()))


def run(label, make, Xtr, ytr, Xte, yte, fit_repeats, pred_repeats):
    fit_times, model = [], None
    for _ in range(fit_repeats):
        model = make()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            fit_times.append(time.perf_counter() - t0)
    converged = not any("converge" in str(w.message) for w in caught)
    pred_times, pred = timed(lambda: model.predict(Xte), pred_repeats)
    proba_times, proba = timed(lambda: model.predict_proba(Xte), pred_repeats)
    p_true = np.clip(proba[np.arange(len(yte)), np.searchsorted(model.classes_, yte)], 1e-15, 1)
    obj, grad = objective(model, Xtr, ytr)
    return {
        "objective": obj,
        "grad_max_abs": grad,
        "label": label,
        "backend": getattr(model, "execution_backend_", "sklearn"),
        "n_iter": int(np.max(model.n_iter_)),
        "converged": converged,
        "fit_s": fit_times,
        "predict_s": pred_times,
        "predict_proba_s": proba_times,
        "train_acc": float((model.predict(Xtr) == ytr).mean()),
        "test_acc": float((pred == yte).mean()),
        "test_logloss": float(-np.log(p_true).mean()),
        "pred": pred.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--fit-repeats", type=int, default=3)
    ap.add_argument("--pred-repeats", type=int, default=20)
    ap.add_argument("--only", choices=["mlx", "sklearn"], default=None)
    ap.add_argument("--dataset", choices=sorted(SPLITS), default="MNIST")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load(args.dataset)
    # Same preprocessing for both: pixels scaled to [0, 1].
    Xtr64, Xte64 = Xtr / 255.0, Xte / 255.0
    Xtr32, Xte32 = Xtr64.astype(np.float32), Xte64.astype(np.float32)
    kw = {"C": args.C, "max_iter": args.max_iter}

    results = []
    if args.only in (None, "mlx"):
        import mlx.core as mx

        from mlxlearn.linear_model import LogisticRegression as MLXLR

        # One untimed warm-up fit so Metal kernel compilation is not billed to the timed fits.
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MLXLR(**kw).fit(Xtr32, ytr)
        cold = time.perf_counter() - t0
        r = run("mlx-learn (float32 input)", lambda: MLXLR(**kw), Xtr32, ytr, Xte32, yte,
                args.fit_repeats, args.pred_repeats)
        r["cold_first_fit_s"] = cold
        results.append(r)
        results.append(run("mlx-learn (float64 input)", lambda: MLXLR(**kw), Xtr64, ytr, Xte64, yte,
                           args.fit_repeats, args.pred_repeats))
        mx.clear_cache()
        time.sleep(2)

    if args.only in (None, "sklearn"):
        from sklearn.linear_model import LogisticRegression as SKLR

        results.append(run("scikit-learn (float64 input)", lambda: SKLR(**kw), Xtr64, ytr, Xte64, yte,
                           args.fit_repeats, args.pred_repeats))

    import mlx.core as mx
    import sklearn
    import threadpoolctl

    meta = {
        "dataset": args.dataset, "max_iter": args.max_iter, "C": args.C,
        "shapes": {"train": list(Xtr.shape), "test": list(Xte.shape)},
        "versions": {"sklearn": sklearn.__version__, "mlx": mx.__version__, "numpy": np.__version__},
        "blas": [{k: i.get(k) for k in ("internal_api", "num_threads")} for i in threadpoolctl.threadpool_info()],
    }
    write_json(args.out, {"meta": meta, "results": results})
    for r in results:
        print(f"{r['label']:30s} backend={r['backend']:7s} iters={r['n_iter']:4d} conv={r['converged']!s:5s} "
              f"fit med={np.median(r['fit_s']):7.3f}s  pred med={1e3*np.median(r['predict_s']):7.2f}ms  "
              f"proba med={1e3*np.median(r['predict_proba_s']):7.2f}ms  "
              f"train={r['train_acc']:.4f} test={r['test_acc']:.4f} ll={r['test_logloss']:.4f} "
              f"obj={r['objective']:.6f} |g|={r['grad_max_abs']:.2e}"
              + (f"  cold={r['cold_first_fit_s']:.3f}s" if "cold_first_fit_s" in r else ""))


if __name__ == "__main__":
    main()
