"""Paired low-dimensional scheduling sweep with sustained GPU warmup."""

import hashlib
import json
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_assignment as exp
from gap_baseline import production as prod

configs = [
    exp.Config(bm=32, bn=64, bk=16),
    exp.Config(bm=64, bn=64, bk=32),
    exp.Config(bm=64, bn=64, bk=32, inner_unroll=False),
    exp.Config(bm=128, bn=64, bk=32, barriers=1),
]
configs += [
    exp.Config(bm=bm, bn=bn, bk=bk, padding=pad, inner_unroll=unroll)
    for bm, bn, bk in [(32, 64, 96), (64, 64, 96), (32, 64, 128), (64, 64, 32)]
    for pad in [0, 4, 8]
    for unroll in [True, False]
]
rng = np.random.default_rng(217)
result = {
    "script": Path(__file__).read_text(),
    "experiment_script": Path(exp.__file__).read_text(),
    "production_sha256": hashlib.sha256(Path(prod.__file__).read_bytes()).hexdigest(),
    "records": [],
    "protocol": "BF16 resident operands; complete warmup >=40ms per method; three interleaved rounds of baseline then candidate, each five synchronized repeats. Median across all 15 repeats. Full generated source preserved per candidate.",
}
out = Path("benchmarks/e_means/results/gap_investigation/lowdim.json")
for d in [96, 100, 128, 200]:
    n, k = 32768, 16384
    x = mx.array(rng.standard_normal((n, d), dtype=np.float32) / np.sqrt(d)).astype(mx.bfloat16)
    c, norm = prod.prepare_centers(
        mx.array(rng.standard_normal((k, d), dtype=np.float32) / np.sqrt(d))
    )
    mx.eval(x, c, norm)
    cn = np.asarray(c.astype(mx.float32)).astype(np.float64)
    xn = np.asarray(x[:19].astype(mx.float32)).astype(np.float64)
    oracle = np.argmin(np.sum(cn * cn, axis=1) - 2 * xn @ cn.T, axis=1)
    baseline = partial(prod.assign_bf16, x, c, norms=norm)
    for conf in configs:
        try:
            conf.validate()
        except AssertionError:
            continue
        if d > 128 and conf.bk > 32:
            continue
        candidate = partial(exp.assign, x, c, norm, conf)
        for fun in [baseline, candidate]:
            mx.eval(fun())
            t = time.perf_counter()
            while time.perf_counter() - t < 0.04:
                mx.eval(fun())
        row = {
            "n": n,
            "k": k,
            "d": d,
            "config": asdict(conf),
            "baseline_seconds": [],
            "candidate_seconds": [],
            "source": exp.kernel_source(conf),
        }
        for _ in range(3):
            for key, fun in [("baseline_seconds", baseline), ("candidate_seconds", candidate)]:
                times, outputs = exp.timed(fun, 5)
                row[key] += times
        row["label_mismatch_first19"] = int(np.count_nonzero(np.asarray(outputs[0][:19]) != oracle))
        assert row["label_mismatch_first19"] == 0
        row["speedup"] = float(
            np.median(row["baseline_seconds"]) / np.median(row["candidate_seconds"])
        )
        result["records"].append(row)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(
            d,
            asdict(conf),
            round(row["speedup"], 3),
            round(np.median(row["baseline_seconds"]) * 1000, 3),
            round(np.median(row["candidate_seconds"]) * 1000, 3),
            flush=True,
        )
