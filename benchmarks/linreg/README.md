# LinearRegression benchmarks

Drivers used for the PR #4 review and rewrite
([docs/linear-regression-2026-11-22.md](../../docs/linear-regression-2026-11-22.md)).
Run them from a checkout with `PYTHONPATH=src`.

| Script | What it measures |
|---|---|
| `bench_linreg.py` | `fit` time and correctness (objective, coefficients, `rank_`, `singular_`) of mlxlearn's direct and patched classes against scikit-learn on three grids: `core` (Gaussian sizes), `hard` (offsets, condition numbers 1e6/1e10/1e14, exact rank loss, one-hot with the intercept, mixed scales, integer data) and `options` (weights, no intercept, ten targets, float32, wide and tiny fits). Each config times every variant in rotated order with a pause before each call, so BLAS threads go cold as in interleaved use. Results append to JSONL and resume by key. |
| `crossover.py` | The patched-dispatch crossover: for every shape, the *slowest* data kind (Gaussian, one-hot, mixed scales, cond 1e10 — one per solver path) against scikit-learn. Patched dispatch may only take shapes where the worst kind still wins. |
| `analyze_xo.py` | Worst-case speedup matrix from `crossover.py` output. |
| `compare_tables.py` | Markdown table of two `bench_linreg.py` runs side by side. |
| `summarize_campaign.py` | Per data kind and option geometric means and minima of a `bench_linreg.py` run (and a PR #4 baseline), plus every cell below parity. |
| `calibrate_routing.py` | Scores candidate patched-routing floors against one or more `crossover.py` runs: shapes routed, and routed shapes where the slowest kind loses. |

Typical use:

```bash
PYTHONPATH=src python benchmarks/linreg/bench_linreg.py hard --out hard.jsonl --label new --variants sk mlx mlx_patched
PYTHONPATH=src python benchmarks/linreg/crossover.py tall --out xo_tall.jsonl
python benchmarks/linreg/analyze_xo.py xo_tall.jsonl
```

Measure on a quiet machine: other processes, and laptops that throttle under
sustained load, move these numbers by more than the margins they decide.
