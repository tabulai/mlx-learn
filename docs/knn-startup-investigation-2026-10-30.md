# KNN total-startup investigation — October 30, 2026

This document records the investigation before the user explicitly authorized
the standalone estimator design below: “Prioritize startup speed; allow that
subclass-identity change.” The rejected import probes remain valid historical
evidence; the authorized architecture and its validation are a subsequent
revision.

The implemented adapter and its measurements are described in the
[startup-fix report](knn-startup-fix-2026-10-30.md).

The tested import rearrangements did not produce a repeatable total-startup
improvement. They were rejected, and the production package initializers were
restored exactly to the preceding frozen implementation. At that stage, no
estimator inheritance change had been implemented.

## Retained experiments

The [package-export probe](../benchmarks/results/knn-startup-2026-10-30-package-comparison.json)
retains 36 fresh-process launches: eleven paired nearest-neighbor runs and seven
paired classifier runs. Lazy root/common exports avoided loading unused RNG and
patching wrappers; sklearn still needed the hashing dependency later. Initial
median import-plus-fit/query improvements were about
3.5 ms and 4.5 ms, respectively, but nearest-neighbor pair signs were mixed.
An 85.6 ms first-pair difference remains in the record; it was not discarded.

The [follow-up ordering probe](../benchmarks/results/knn-startup-2026-10-30-order-probe.json)
retains another 72 launches: Iris and Wine, nine repetitions each, with preceding
imports, lazy package exports, lazy exports plus earlier native loading, and
sklearn auto. Estimator setup plus first fit/query medians were:

| Fixture | Preceding imports | Lazy packages | Earlier native loading | sklearn auto |
|---|---:|---:|---:|---:|
| Iris | 749.83 ms | 755.69 ms | 759.76 ms | 734.77 ms |
| Wine | 770.74 ms | 768.54 ms | 768.49 ms | 757.04 ms |

The sign and size of the changes depended on the fixture and launch. Neither
candidate established the required startup improvement. These probes used
synthetic nearest-neighbor/classifier data and Iris/Wine splits; they are
diagnostics, not replacements for the full benchmark panel.

All **108 launches** remain available. Runs were serial and interleaved in a
seeded shuffled order. Separate copied source roots had identical native binary
bytes and equivalent initial bytecode-cache state; their complete source
inventories and exact differing paths are recorded. Both artifacts include the
reproducible driver. First launches, setup, first fit/query and combined timings
remain visible; the second probe also records parent process-to-result latency.
There was no prewarm, background loading or timer-specific estimator hook.
These are fresh-process measurements, not an OS-cold-cache claim.

## Import dependency and compatibility boundary

The [current import trace](../benchmarks/results/knn-startup-2026-10-30-importtime-current.log)
attributes about 621 ms to the sklearn root import and about 750 ms cumulatively
to `sklearn.neighbors`: roughly **129 ms** beyond the root. The additional branch
includes neighbor trees, metrics/pairwise extensions and other neighbor
estimators. The [sklearn trace](../benchmarks/results/knn-startup-2026-10-30-importtime-sklearn.log)
provides the corresponding baseline. The native CPU extension itself took about
2.8 ms to load in the current trace. Moving that load earlier did not establish
a total-latency benefit.

The 129 ms difference is attribution from **one instrumented import profile**,
not a validated speedup prediction. A redesigned estimator would still need
public `sklearn.base`, validation helpers, NumPy, SciPy and its own search
dependencies; its actual total must be measured.

The historical neighbor contract in the
[preceding frozen source](../benchmarks/reference/knn-latency-final-2026-10-30.zip),
`mlxlearn/neighbors/_base.py`, said:

> Each mlxlearn estimator subclasses its scikit-learn counterpart.

It explicitly connected that inheritance to constructor signatures, parameter
constraints, tags, cloning and `isinstance`. For example, this held then
against a stock class captured before patching:

```python
from sklearn.neighbors import NearestNeighbors as StockNeighbors
from mlxlearn.neighbors import NearestNeighbors

assert isinstance(NearestNeighbors(), StockNeighbors)
```

## Alternative proposed before approval

A standalone estimator based on public `BaseEstimator` and the public
classifier/regressor mixins could avoid importing `sklearn.neighbors` for the
owned Euclidean path. A concrete design would:

- Preserve explicit constructor signatures, parameters, tags, metadata routing,
  cloning, scoring, serialization and the existing numerical search.
- Validate a proven-valid common parameter subset locally, with stock parameter
  validation loaded on other cases to preserve version-specific errors.
- Build neighbor graphs from owned results, and import stock neighbors only
  for radius operations or an explicitly permitted capability fallback.
- Capture original sklearn classes before patching to prevent recursive
  delegation, and preserve fitted-state, ownership and diagnostic behavior.

This can preserve sklearn estimator compatibility, but it **cannot preserve
genuine stock-class inheritance for direct estimators** while avoiding that
class's import. The `isinstance` assertion above would change. Patched classes
could retain stock inheritance because patching already imports those classes;
that would not preserve the direct-estimator contract. Approval of that tradeoff
and broad constructor, validation, tags, metadata, graph/radius, fallback and
serialization tests were needed before adopting the design. The user later
approved the tradeoff, and the subsequent implementation preserves actual stock
inheritance for patched classes while changing the direct classes.

## Diagnostics test correction

The investigation also exposed a stale assertion in
[the diagnostics test](../tests/unit/test_diagnostics.py#L104). It expected the
old fit-crossover detail after automatic CPU/GPU dispatch had replaced that
decision. The corrected test explicitly selects automatic mode and checks the
recorded shape, CPU storage backend, `auto-fit-storage` reason and absence of
sklearn fallback. This preserves the original auditability intent without
changing production dispatch.
