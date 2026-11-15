"""Validate updated production BF16 dispatch against frozen v2 and FP64 oracle."""

import hashlib
import json
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np
from gap_baseline import production as old

from mlxlearn._kernels import e_means_bf16 as new

rng = np.random.default_rng(409)
result = {
    "script": Path(__file__).read_text(),
    "old_source_sha256": hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
    "new_source_sha256": hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest(),
    "correctness": [],
    "timing": [],
}
output = Path("benchmarks/e_means/results/gap_investigation/production-validation.json")
for n, k, d in [
    (1, 1, 96),
    (7, 3, 128),
    (33, 65, 96),
    (67, 129, 128),
    (131, 257, 128),
    (65, 71, 100),
    (35, 65, 1024),
]:
    x = mx.array(rng.standard_normal((n, d), dtype=np.float32)).astype(mx.bfloat16)
    c, norms = new.prepare_centers(mx.array(rng.standard_normal((k, d), dtype=np.float32)))
    xn = np.asarray(x.astype(mx.float32)).astype(np.float64)
    cn = np.asarray(c.astype(mx.float32)).astype(np.float64)
    for spherical in [False, True]:
        labels, scores = new.assign_bf16(x, c, norms=norms, batch_size=17, spherical=spherical)
        oldlabels, oldscores = old.assign_bf16(
            x, c, norms=norms, batch_size=17, spherical=spherical
        )
        values = -(xn @ cn.T) if spherical else np.sum(cn * cn, axis=1) - 2 * (xn @ cn.T)
        expected = values.argmin(axis=1)
        np.testing.assert_array_equal(np.asarray(labels), expected)
        np.testing.assert_array_equal(np.asarray(labels), np.asarray(oldlabels))
        np.testing.assert_allclose(
            np.asarray(scores), values[np.arange(n), expected], rtol=3e-6, atol=3e-4
        )
        result["correctness"].append(
            {
                "n": n,
                "k": k,
                "d": d,
                "spherical": spherical,
                "oracle_label_mismatches": 0,
                "frozen_label_mismatches": 0,
                "maximum_score_difference": float(
                    np.max(abs(np.asarray(scores) - np.asarray(oldscores)))
                ),
            }
        )
for d in [96, 128]:
    x = mx.full((11, d), 100000, mx.float32)
    c = mx.full((131, d), 100000, mx.float32)
    labels, scores = new.assign_bf16(x, c)
    np.testing.assert_array_equal(np.asarray(labels), np.zeros(11, np.uint32))
    assert np.isfinite(np.asarray(scores)).all()
result["duplicate_ties_beyond_fp16_range_pass"] = True
for d, n, k in [
    (96, 32768, 16384),
    (128, 32768, 16384),
    (96, 8192, 8192),
    (128, 8192, 8192),
    (100, 8192, 16384),
    (1024, 8192, 16384),
]:
    x = mx.array(rng.standard_normal((n, d), dtype=np.float32) / np.sqrt(d)).astype(mx.bfloat16)
    c, norms = new.prepare_centers(
        mx.array(rng.standard_normal((k, d), dtype=np.float32) / np.sqrt(d))
    )
    mx.eval(x, c, norms)
    methods = [
        ("old", partial(old.assign_bf16, x, c, norms=norms)),
        ("new", partial(new.assign_bf16, x, c, norms=norms)),
    ]
    for _name, fun in methods:
        mx.eval(fun())
        started = time.perf_counter()
        while time.perf_counter() - started < 0.08:
            mx.eval(fun())
    record = {"n": n, "k": k, "d": d, "old_seconds": [], "new_seconds": []}
    outputs = {}
    for r in range(5):
        for name, fun in methods if r % 2 == 0 else methods[::-1]:
            for _repeat in range(7):
                started = time.perf_counter()
                out = fun()
                mx.eval(out)
                mx.synchronize()
                record[name + "_seconds"].append(time.perf_counter() - started)
            outputs[name] = out
    np.testing.assert_array_equal(np.asarray(outputs["new"][0]), np.asarray(outputs["old"][0]))
    record["all_label_mismatches"] = 0
    record["speedup"] = float(np.median(record["old_seconds"]) / np.median(record["new_seconds"]))
    result["timing"].append(record)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        d,
        n,
        k,
        "speedup",
        round(record["speedup"], 4),
        "old_ms",
        round(np.median(record["old_seconds"]) * 1000, 4),
        "new_ms",
        round(np.median(record["new_seconds"]) * 1000, 4),
        flush=True,
    )
print("correctness passed", len(result["correctness"]), flush=True)
