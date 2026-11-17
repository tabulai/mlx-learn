"""Synchronized host dispatch latency and device FP32-copy bandwidth diagnostic."""

import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

kernel = mx.fast.metal_kernel(
    name="emeans_gap_copy",
    input_names=["x"],
    output_names=["y"],
    source="uint i=thread_position_in_grid.x; if(i<N)y[i]=x[i];",
)
records = []
for n, repeats in [(1, 200), (65536, 100), (2**24, 30), (2**27, 10)]:
    x = mx.ones((n,), mx.float32)
    mx.eval(x)

    def invoke(x=x, n=n):
        return kernel(
            inputs=[x],
            template=[("N", n)],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n,)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )[0]

    mx.eval(invoke())
    start = time.perf_counter()
    while time.perf_counter() - start < 0.05:
        mx.eval(invoke())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        y = invoke()
        mx.eval(y)
        mx.synchronize()
        times.append(time.perf_counter() - start)
    assert float(y[0].item()) == 1 and float(y[-1].item()) == 1
    records.append(
        {
            "elements": n,
            "array_bytes": 4 * n,
            "traffic_bytes_read_plus_write": 8 * n,
            "seconds": times,
            "median_seconds": float(np.median(times)),
            "read_plus_write_GB_per_s": float(8 * n / np.median(times) / 1e9),
        }
    )
    print(
        records[-1]["elements"],
        records[-1]["median_seconds"],
        records[-1]["read_plus_write_GB_per_s"],
        flush=True,
    )
script = Path(__file__).read_text()
Path("benchmarks/e_means/results/gap_investigation/dispatch-bandwidth.json").write_text(
    json.dumps(
        {
            "script": script,
            "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
            "hardware": mx.device_info(),
            "protocol": "50ms sustained warmup, isolated synchronized dispatches; copy bandwidth counts both device read and write; singleton measures end-to-end Python+MLX+driver+GPU latency, not bare hardware launch latency",
            "records": records,
        },
        indent=2,
    )
    + "\n"
)
