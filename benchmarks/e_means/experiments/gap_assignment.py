"""Reproducible BF16 scheduling/tiling experiments; no runtime mutations.

Use --roofline for diagnostic MLX matmuls. FP16 and BF16-output matmuls in
that diagnostic are not valid substitutes for BF16-input/FP32-output assignment.
Kernel source derives from the attributed production implementation; each
result records the exact generated source, source hashes, and configurations.

Copyright © 2024 Apple Inc.
Modified for mlxlearn on 2026-11-14: generate scheduling and tiling variants
of the attributed MLX-derived Metal kernel; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from pathlib import Path

import mlx.core as mx
import numpy as np
from gap_baseline import production


@dataclass(frozen=True)
class Config:
    bm: int = 64
    bn: int = 64
    bk: int = 32
    wm: int = 2
    wn: int = 2
    padding: int = 8
    barriers: int = 0
    reverse_mma: bool = False
    inner_unroll: bool = True

    def validate(self):
        assert min(self.bm, self.bn, self.bk, self.wm, self.wn) > 0
        assert self.bm % (8 * self.wm) == self.bn % (8 * self.wn) == self.bk % 8 == 0
        threads = self.wm * self.wn * 32
        assert threads <= 1024
        for rows in [self.bm, self.bn]:
            vector = rows * self.bk // threads
            assert vector > 0 and self.bk % vector == 0
        shared = (self.bm + self.bn) * (self.bk + self.padding) * 2 + 8 * self.wn * self.bm
        assert shared <= 32768, shared
        return threads


HEADER = r"""
#include <metal_simdgroup_matrix>
template <typename V, typename U>
V emeans_bf16_load_vector(const device U* pointer) {
    return *reinterpret_cast<const device V*>(pointer);
}
template <typename V, typename U>
V emeans_bf16_load_vector(const constant U* pointer) {
    return *reinterpret_cast<const constant V*>(pointer);
}
"""


def kernel_source(config):
    source = production._TILE_SOURCE
    source = source.replace(
        "metal::simdgroup_matrix<T,8,8> aa[TM], bb[TN];",
        "if constexpr (BARRIERS == 1 || BARRIERS == 3) metal::simdgroup_barrier(mem_flags::mem_none);\n"
        "            metal::simdgroup_matrix<T,8,8> aa[TM], bb[TN];",
    )
    source = source.replace(
        "            for (uint n = 0; n < TN; ++n) {\n                if constexpr (MANUAL)",
        "            for (uint n = 0; n < TN; ++n) {\n                if constexpr (MANUAL)",
        1,
    )
    before_b = "            #pragma clang loop unroll(full)\n            for (uint n = 0; n < TN; ++n) {\n                if constexpr (MANUAL)"
    source = source.replace(
        before_b,
        "            if constexpr (BARRIERS == 1) metal::simdgroup_barrier(mem_flags::mem_none);\n"
        + before_b,
    )
    mma = """            #pragma clang loop unroll(full)
            for (uint m = 0; m < TM; ++m)
                #pragma clang loop unroll(full)
                for (uint n = 0; n < TN; ++n)
                    metal::simdgroup_multiply_accumulate(acc[m][n], aa[m], bb[n], acc[m][n]);"""
    replacement = mma
    if config.reverse_mma:
        replacement = (
            mma.replace(
                "uint m = 0; m < TM; ++m", "uint placeholder = 0; placeholder < TN; ++placeholder"
            )
            .replace("uint n = 0; n < TN; ++n", "uint m = 0; m < TM; ++m")
            .replace(
                "uint placeholder = 0; placeholder < TN; ++placeholder", "uint n = 0; n < TN; ++n"
            )
        )
    source = source.replace(
        mma,
        "            if constexpr (BARRIERS == 1 || BARRIERS == 2) metal::simdgroup_barrier(mem_flags::mem_none);\n"
        + replacement,
    )
    if not config.inner_unroll:
        source = source.replace(
            "#pragma clang loop unroll(full)\n        for (uint f = 0;",
            "#pragma clang loop unroll(disable)\n        for (uint f = 0;",
        )
    return source


@lru_cache(None)
def kernel(config):
    config.validate()
    return mx.fast.metal_kernel(
        name="emeans_gap_scheduling",
        input_names=["x", "centers", "norms"],
        output_names=["partial_scores", "partial_indices"],
        source=kernel_source(config),
        header=HEADER,
        compile_options={"math_mode": "safe"},
    )


def assign(x, centers, norms, config):
    n, d = x.shape
    k = centers.shape[0]
    parts = (k + config.bn - 1) // config.bn
    threads = config.validate()
    partial = kernel(config)(
        inputs=[x, centers, norms],
        template=[
            ("T", mx.bfloat16),
            ("N", n),
            ("K", k),
            ("D", d),
            ("BM", config.bm),
            ("BN", config.bn),
            ("BK", config.bk),
            ("WM", config.wm),
            ("WN", config.wn),
            ("PAD", config.padding),
            ("PARTS", parts),
            ("MANUAL", True),
            ("VECTOR_LOAD", True),
            ("NATIVE", True),
            ("SPHERICAL", False),
            ("BARRIERS", config.barriers),
        ],
        grid=(((n + config.bm - 1) // config.bm) * threads, parts, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(n, parts), (n, parts)],
        output_dtypes=[mx.float32, mx.uint32],
        stream=mx.gpu,
    )
    return tuple(
        production._merge_kernel()(
            inputs=partial,
            template=[("N", n), ("PARTS", parts)],
            grid=(n * 32, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.uint32, mx.float32],
            stream=mx.gpu,
        )
    )


def timed(fun, repetitions):
    out = fun()
    mx.eval(out)
    mx.synchronize()
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        out = fun()
        mx.eval(out)
        mx.synchronize()
        timings.append(time.perf_counter() - started)
    return timings, out


def configurations(stage):
    if stage == "schedule":
        return [
            Config(barriers=b, reverse_mma=reverse, inner_unroll=unroll)
            for b in range(4)
            for reverse, unroll in [(False, True), (True, True), (False, False)]
        ]
    if stage == "tiles":
        return [
            Config(bm=bm, bn=bn, bk=bk, wm=wm, wn=wn, barriers=barriers)
            for bm, bn, wm, wn in [
                (32, 64, 2, 2),
                (64, 64, 2, 2),
                (64, 128, 2, 4),
                (128, 64, 4, 2),
                (64, 128, 2, 2),
                (128, 64, 2, 2),
                (128, 128, 4, 2),
            ]
            for bk in [16, 32, 64]
            for barriers in [0, 1]
        ]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roofline", action="store_true")
    parser.add_argument("--stage", choices=["schedule", "tiles"], default="schedule")
    parser.add_argument("--dims", default="96,1024,3072")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--clusters", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    script = Path(__file__).read_text()
    source = Path(production.__file__).read_text()
    result = {
        "protocol": "resident random normal / sqrt(d) represented BF16 matrices; preparation excluded; one warmup, synchronized repeats; sample labels checked against CPU FP64 represented-value oracle",
        "command_args": vars(args),
        "hardware": mx.device_info(),
        "script": script,
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "production_source": source,
        "production_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "records": [],
    }
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    def save():
        outpath.write_text(json.dumps(result, indent=2) + "\n")

    rng = np.random.default_rng(701)
    for d in map(int, args.dims.split(",")):
        n, k = args.rows, args.clusters
        x = mx.array(rng.standard_normal((n, d), dtype=np.float32) / np.sqrt(d)).astype(mx.bfloat16)
        centers, norms = production.prepare_centers(
            mx.array(rng.standard_normal((k, d), dtype=np.float32) / np.sqrt(d))
        )
        mx.eval(x, centers, norms)
        cn = np.asarray(centers.astype(mx.float32)).astype(np.float64)
        xn = np.asarray(x[:19].astype(mx.float32)).astype(np.float64)
        oracle = np.argmin(np.sum(cn * cn, axis=1) - 2 * (xn @ cn.T), axis=1)
        methods = [("production", partial(production.assign_bf16, x, centers, norms=norms), None)]
        if args.roofline:
            for dtype, name in [
                (mx.float16, "diagnostic_fp16_output"),
                (mx.bfloat16, "diagnostic_bf16_output"),
                (mx.float32, "represented_fp32_output"),
            ]:
                a, b = x.astype(dtype), centers.astype(dtype).T
                mx.eval(a, b)
                methods.append((name, lambda a=a, b=b: a @ b, None))
        else:
            for config in configurations(args.stage):
                try:
                    config.validate()
                except AssertionError:
                    continue
                methods.append(
                    ("candidate", partial(assign, x, centers, norms, config), config)
                )
        for name, fun, config in methods:
            row = {
                "n": n,
                "k": k,
                "d": d,
                "method": name,
                "config": asdict(config) if config else None,
            }
            if config:
                row["kernel_source"] = kernel_source(config)
            try:
                seconds, out = timed(fun, args.repeats)
                median = float(np.median(seconds))
                row.update(
                    seconds=seconds, median_seconds=median, dot_tflops=2 * n * k * d / median / 1e12
                )
                if name in ["production", "candidate"]:
                    row["oracle_mismatches_first19"] = int(
                        np.count_nonzero(np.asarray(out[0][:19]) != oracle)
                    )
                    if row["oracle_mismatches_first19"]:
                        raise AssertionError(f"oracle mismatch: {row['oracle_mismatches_first19']}")
                else:
                    row["diagnostic_only"] = (
                        "matmul output precision differs or excludes argmin; not a production assignment substitute"
                    )
            except Exception as exc:
                row["error"] = str(exc)
            result["records"].append(row)
            print(
                d,
                name,
                asdict(config) if config else "",
                row.get("median_seconds"),
                row.get("error", ""),
                flush=True,
            )
            save()


if __name__ == "__main__":
    main()
