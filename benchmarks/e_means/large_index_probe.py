"""Verify DINO10M gathers across 32-bit element offsets before benchmarking."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402
from mlxlearn._kernels.e_means import gather_bf16  # noqa: E402


def main():
    data, _, _ = base.reference_modules(base.REPO / "tmp/e_means/faiss", base.DEFAULT_CACHE)
    x, _, metadata = base.load_dataset("dino10b-10M-div4096", data, base.DEFAULT_CACHE)
    indices = np.array([0, 4_194_303, 4_194_304, 8_388_607, 8_388_608, 9_999_999], np.int32)
    expected = x[indices].copy()
    resident = mx.array(x)
    mx.eval(resident)
    del x
    gc.collect()
    order = mx.array(indices)
    fp32 = np.asarray(resident[order])
    bf16 = np.asarray(gather_bf16(resident, order).astype(mx.float32))
    expected_bf16 = np.asarray(mx.array(expected).astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_array_equal(fp32, expected)
    np.testing.assert_array_equal(bf16, expected_bf16)
    result = {
        "status": "passed", "n": 10_000_000, "d": 1024,
        "indices": indices.tolist(), "element_offsets": (indices.astype(np.int64) * 1024).tolist(),
        "fp32_exact": True, "bf16_exact": True, "dataset": metadata,
        "probe_source_sha256": base.file_sha256(__file__),
        "runtime_source_sha256": {
            str(path.relative_to(base.REPO)): base.file_sha256(path)
            for path in (base.REPO / "src/mlxlearn/_kernels/e_means.py",
                         base.REPO / "src/mlxlearn/_kernels/e_means_bf16.py")
        },
        "versions": base.package_versions(), "hardware": base.host_identity(),
        "expected_fp32_sha256": base.array_sha256(expected),
        "expected_bf16_as_fp32_sha256": base.array_sha256(expected_bf16),
        "scope": "Correctness diagnostic only; excluded from all performance and memory results.",
    }
    destination = HERE / "results/large_validation/dino-index-boundaries.json"
    base.atomic_json(destination, result)
    print(f"PASS: FP32 and BF16 gathers beyond 2**32 elements; {destination}", flush=True)


if __name__ == "__main__":
    main()
