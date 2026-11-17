"""Coalesced partial argmin writes and coalesced parallel merge experiments.

Copyright © 2024 Apple Inc.
Modified for mlxlearn on 2026-11-14: change partial-result layout in the
attributed MLX-derived Metal kernel; see THIRD_PARTY_NOTICES.md.
"""

import hashlib
import json
import time
from functools import lru_cache, partial
from pathlib import Path

import mlx.core as mx
import numpy as np
from gap_baseline import production as prod

HEADER = r"""
#include <metal_simdgroup_matrix>
template<typename V,typename U> V emeans_bf16_load_vector(const device U* p){return *reinterpret_cast<const device V*>(p);}
template<typename V,typename U> V emeans_bf16_load_vector(const constant U* p){return *reinterpret_cast<const constant V*>(p);}
"""
SOURCE = prod._TILE_SOURCE.replace("[(r0 + row) * PARTS + c0 / BN]", "[(c0 / BN) * N + r0 + row]")
MERGE = r"""
uint lane=thread_index_in_simdgroup;
uint warp=simdgroup_index_in_threadgroup;
uint tid=thread_index_in_threadgroup;
uint row=threadgroup_position_in_grid.x*32+lane;
float best=INFINITY;
uint index=0;
for(uint part=warp;part<PARTS;part+=WARPS){
    float score=row<N?scores[part*N+row]:INFINITY;
    uint candidate=row<N?indices[part*N+row]:0;
    if(score<best||(score==best&&candidate<index)){best=score;index=candidate;}
}
threadgroup float ss[WARPS*32];
threadgroup uint ii[WARPS*32];
ss[tid]=best;ii[tid]=index;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(warp==0&&row<N){
    for(uint w=1;w<WARPS;++w){
        float value=ss[w*32+lane];uint candidate=ii[w*32+lane];
        if(value<best||(value==best&&candidate<index)){best=value;index=candidate;}
    }
    labels[row]=index;best_scores[row]=best;
}
"""


@lru_cache(None)
def kernels():
    return (
        mx.fast.metal_kernel(
            name="emeans_gap_coalesced_tiles",
            input_names=["x", "centers", "norms"],
            output_names=["partial_scores", "partial_indices"],
            source=SOURCE,
            header=HEADER,
            compile_options={"math_mode": "safe"},
        ),
        mx.fast.metal_kernel(
            name="emeans_gap_coalesced_merge",
            input_names=["scores", "indices"],
            output_names=["labels", "best_scores"],
            source=MERGE,
            compile_options={"math_mode": "safe"},
        ),
    )


def assign(x, c, norms, warps, bm=None):
    n, d = x.shape
    k = c.shape[0]
    bm, bn, bk = ((32, 64, 16) if d <= 128 else (64, 64, 32)) if bm is None else (bm, 64, 32)
    parts = (k + bn - 1) // bn
    tile, merge = kernels()
    partial = tile(
        inputs=[x, c, norms],
        template=[
            ("T", mx.bfloat16),
            ("N", n),
            ("K", k),
            ("D", d),
            ("BM", bm),
            ("BN", bn),
            ("BK", bk),
            ("WM", 2),
            ("WN", 2),
            ("PAD", 8),
            ("PARTS", parts),
            ("MANUAL", True),
            ("VECTOR_LOAD", True),
            ("NATIVE", True),
            ("SPHERICAL", False),
        ],
        grid=(((n + bm - 1) // bm) * 128, parts, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(parts, n), (parts, n)],
        output_dtypes=[mx.float32, mx.uint32],
        stream=mx.gpu,
    )
    return tuple(
        merge(
            inputs=partial,
            template=[("N", n), ("PARTS", parts), ("WARPS", warps)],
            grid=(((n + 31) // 32) * 32 * warps, 1, 1),
            threadgroup=(32 * warps, 1, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.uint32, mx.float32],
            stream=mx.gpu,
        )
    )


def main():
    rng = np.random.default_rng(302)
    result = {
        "script": Path(__file__).read_text(),
        "tile_source": SOURCE,
        "merge_source": MERGE,
        "production_sha256": hashlib.sha256(Path(prod.__file__).read_bytes()).hexdigest(),
        "protocol": "resident BF16, FP32 norms and score accumulation, 50ms sustained warmup; three alternating baseline/candidate rounds with seven synchronized repetitions per round; sample19 CPU FP64 represented assignment oracle",
        "records": [],
    }
    out = Path("benchmarks/e_means/results/gap_investigation/coalescing.json")
    for d in [96, 100, 128, 200, 1024]:
        n = 32768 if d <= 200 else 8192
        k = 16384
        x = mx.array(rng.standard_normal((n, d), dtype=np.float32) / np.sqrt(d)).astype(mx.bfloat16)
        c, norms = prod.prepare_centers(
            mx.array(rng.standard_normal((k, d), dtype=np.float32) / np.sqrt(d))
        )
        mx.eval(x, c, norms)
        cn = np.asarray(c.astype(mx.float32)).astype(np.float64)
        xn = np.asarray(x[:19].astype(mx.float32)).astype(np.float64)
        oracle = np.argmin(np.sum(cn * cn, axis=1) - 2 * xn @ cn.T, axis=1)
        baseline = partial(prod.assign_bf16, x, c, norms=norms)
        for warps, bm in [(1, None), (4, None), (8, None), (16, None), (8, 64)]:
            candidate = partial(assign, x, c, norms, warps, bm)
            row = {
                "n": n,
                "k": k,
                "d": d,
                "merge_warps": warps,
                "override_bm": bm,
                "baseline_seconds": [],
                "candidate_seconds": [],
            }
            for fun in [baseline, candidate]:
                mx.eval(fun())
                t = time.perf_counter()
                while time.perf_counter() - t < 0.05:
                    mx.eval(fun())
            for _ in range(3):
                for key, fun in [("baseline_seconds", baseline), ("candidate_seconds", candidate)]:
                    for _repeat in range(7):
                        t = time.perf_counter()
                        outputs = fun()
                        mx.eval(outputs)
                        mx.synchronize()
                        row[key].append(time.perf_counter() - t)
            row["oracle_mismatches_first19"] = int(
                np.count_nonzero(np.asarray(outputs[0][:19]) != oracle)
            )
            assert row["oracle_mismatches_first19"] == 0
            row["speedup"] = float(
                np.median(row["baseline_seconds"]) / np.median(row["candidate_seconds"])
            )
            result["records"].append(row)
            out.write_text(json.dumps(result, indent=2) + "\n")
            print(
                d,
                warps,
                bm,
                round(row["speedup"], 3),
                round(np.median(row["baseline_seconds"]) * 1000, 3),
                round(np.median(row["candidate_seconds"]) * 1000, 3),
                flush=True,
            )


if __name__ == "__main__":
    main()
