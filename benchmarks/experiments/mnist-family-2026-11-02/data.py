"""Train/validation loaders for the MNIST-family datasets, plus result-file helpers.

``MNIST_FAMILY_ROOT`` must name the directory holding ``MNIST/``, ``Fashion_MNIST/``
and ``Kannada_MNIST/``, each with an ``input/`` folder of the Kaggle CSVs. Parsed
pixels are cached as uint8 under ``MNIST_FAMILY_CACHE`` (default
``~/.cache/mlxlearn/mnist-family``).
"""

import gzip
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
# Kannada's test.csv.gz is the unlabeled Kaggle test set (an ``id`` column, no
# ``label``), so its labeled hold-out Dig-MNIST is the validation set there.
SPLITS = {
    "MNIST": ("train.csv.gz", "test.csv.gz"),
    "Fashion_MNIST": ("train.csv.gz", "test.csv.gz"),
    "Kannada_MNIST": ("train.csv.gz", "Dig-MNIST.csv.gz"),
}


def _parse(name):
    root = os.environ.get("MNIST_FAMILY_ROOT")
    if not root:
        raise SystemExit("Set MNIST_FAMILY_ROOT to the directory containing MNIST/, "
                         "Fashion_MNIST/ and Kannada_MNIST/.")
    arrays = {}
    for split, fname in zip(("tr", "te"), SPLITS[name], strict=True):
        with gzip.open(Path(root) / name / "input" / fname, "rt") as f:
            assert f.readline().split(",")[0] == "label"
            arr = np.loadtxt(f, delimiter=",", dtype=np.float64)
        X = arr[:, 1:]
        assert X.min() >= 0 and X.max() <= 255 and np.all(X == np.round(X))
        arrays["X" + split] = X.astype(np.uint8)
        arrays["y" + split] = arr[:, 0].astype(np.int64)
    return arrays


def load(name):
    """Return raw pixels as integer-valued float64 Xtr, Xte and int64 labels ytr, yte."""
    cache_dir = Path(os.environ.get("MNIST_FAMILY_CACHE",
                                    Path.home() / ".cache" / "mlxlearn" / "mnist-family"))
    cache = cache_dir / f"{name.lower()}.npz"
    if not cache.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, **_parse(name))
        tmp.replace(cache)
    d = np.load(cache)
    return d["Xtr"].astype(np.float64), d["ytr"], d["Xte"].astype(np.float64), d["yte"]


def write_json(path, obj):
    """Write ``obj`` as JSON, gzip-compressed when ``path`` ends in ``.gz``."""
    path, text = Path(path), json.dumps(obj)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())
