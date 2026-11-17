"""mlx-learn vs scikit-learn SVC on one MNIST-family dataset. One library per process."""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=["mlx", "sklearn"], required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--cache-size", type=float, default=200)
    ap.add_argument("--fit-repeats", type=int, default=1)
    ap.add_argument("--pred-repeats", type=int, default=1)
    ap.add_argument("--dataset", choices=sorted(SPLITS), default="MNIST")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load(args.dataset)
    if args.n_train:
        Xtr, ytr = Xtr[:args.n_train], ytr[:args.n_train]
    # Same preprocessing as the other runs: pixels scaled to [0, 1]. Both SVCs
    # validate to float64, so one input dtype covers both.
    Xtr, Xte = Xtr / 255.0, Xte / 255.0
    kw = {"cache_size": args.cache_size}  # everything else at defaults: rbf, C=1, gamma="scale"

    if args.lib == "mlx":
        import mlx.core as mx

        from mlxlearn.svm import SVC

        # Untimed warm-up on a small slice so kernel compilation is not billed to the timed fit.
        t0 = time.perf_counter()
        SVC(**kw).fit(Xtr[:2000], ytr[:2000]).predict(Xte[:500])
        warmup = time.perf_counter() - t0
        versions = {"mlx": mx.__version__}
    else:
        from sklearn.svm import SVC
        warmup = None
        versions = {}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_times, model = timed(lambda: SVC(**kw).fit(Xtr, ytr), args.fit_repeats)
    pred_times, pred = timed(lambda: model.predict(Xte), args.pred_repeats)
    dec_times, dec = timed(lambda: model.decision_function(Xte), args.pred_repeats)
    train_sub = slice(0, 10000)
    train_acc = float((model.predict(Xtr[train_sub]) == ytr[train_sub]).mean())

    import sklearn
    versions.update(sklearn=sklearn.__version__, numpy=np.__version__)
    out = {
        "dataset": args.dataset,
        "lib": args.lib,
        "n_train": len(Xtr),
        "cache_size": args.cache_size,
        "backend": getattr(model, "execution_backend_", "sklearn"),
        "warmup_s": warmup,
        "fit_s": fit_times,
        "predict_s": pred_times,
        "decision_s": dec_times,
        "test_acc": float((pred == yte).mean()),
        "train_acc_first10k": train_acc,
        "n_support": model.n_support_.tolist(),
        "n_iter": np.asarray(getattr(model, "n_iter_", [])).tolist(),
        "fit_status": int(model.fit_status_),
        "gamma": float(model._gamma),
        "warnings": [str(w.message)[:200] for w in caught],
        "pred": pred.tolist(),
        "dec": dec.tolist(),
        "versions": versions,
    }
    write_json(args.out, out)
    print(f"{args.lib:8s} n={len(Xtr):6d} backend={out['backend']:7s} fit={np.median(fit_times):8.2f}s "
          f"predict={np.median(pred_times):7.2f}s decision={np.median(dec_times):7.2f}s "
          f"test={out['test_acc']:.4f} train10k={train_acc:.4f} nSV={sum(out['n_support'])} "
          f"iters={sum(out['n_iter'])} status={out['fit_status']} warn={len(caught)}"
          + (f" warmup={warmup:.2f}s" if warmup is not None else ""))


if __name__ == "__main__":
    main()
