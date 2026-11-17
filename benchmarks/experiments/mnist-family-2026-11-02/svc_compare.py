"""Compare ``svc.py`` result files against the scikit-learn one among them."""

import sys
from pathlib import Path

import numpy as np
from data import read_json


def split_ovr(ovr):
    """scikit-learn's ovr value is votes + s / (3 (|s| + 1)), s = summed pairwise confidence."""
    votes = np.rint(ovr)
    t = ovr - votes
    return votes, 3 * t / (1 - 3 * np.abs(t))


runs = {Path(p).name: read_json(p) for p in sys.argv[1:]}
ref_name = next(p for p, r in runs.items() if r["lib"] == "sklearn")
ref = runs[ref_name]
ref_pred, ref_dec = np.array(ref["pred"]), np.array(ref["dec"])
ref_votes, ref_conf = split_ovr(ref_dec)
for p, r in runs.items():
    pred, dec = np.array(r["pred"]), np.array(r["dec"])
    f, q, dd = np.array(r["fit_s"]), np.array(r["predict_s"]), np.array(r["decision_s"])
    print(f"{p:40s} {r['lib']:7s} cache={r['cache_size']:6.0f} backend={r['backend']:7s} "
          f"fit min/med {f.min():7.2f}/{np.median(f):7.2f}s  predict min/med {q.min():6.2f}/{np.median(q):6.2f}s  "
          f"decision med {np.median(dd):6.2f}s  test={r['test_acc']:.4f} train10k={r['train_acc_first10k']:.4f} "
          f"nSV={sum(r['n_support'])} iters={sum(r['n_iter'])} status={r['fit_status']} gamma={r['gamma']:.6g}")
    votes, conf = split_ovr(dec)
    cd = np.abs(conf - ref_conf)
    print(f"{'':40s} vs {ref_name}: pred agree={np.mean(pred == ref_pred):.4%}  "
          f"rows with a flipped pairwise vote={np.mean(np.any(votes != ref_votes, axis=1)):.3%}  "
          f"summed-confidence |diff| median/max={np.median(cd):.1e}/{cd.max():.1e}  "
          f"nSV/class diff={np.subtract(r['n_support'], ref['n_support']).tolist()}  "
          f"warnings={r['warnings']}")
