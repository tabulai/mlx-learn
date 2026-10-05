# Development

```bash
git clone https://github.com/tabulai/mlx-learn
cd mlx-learn
pip install -e ".[dev]"
pytest
```

`pytest` runs the fast suite. Tests marked `slow` are deselected by default (`addopts` in
`pyproject.toml`); `pytest -m slow` runs just those, and `pytest -m ""` runs everything. CI
runs the slow tests in a job of their own on every PR, so deselecting them locally never
means nobody runs them.

## Layout

```
src/mlxlearn/
├── _common/     configuration, device, arrays, validation, RNG, diagnostics, dispatch
├── _kernels/    MLX array operations; knows nothing about estimators
├── _solvers/    typed config in, typed state out; never sees an estimator or scikit-learn
├── neighbors/ linear_model/ svm/   Layer 1 estimators
└── patching/    Layer 2 — the only package allowed near scikit-learn's namespace
tests/  benchmarks/  docs/  tools/  phase0/
```

The layering is enforced, not merely described:

- A `_solvers` or `_kernels` module that imports `sklearn`, reads the user's configuration
  (`get_config`, `set_config`, `config_context`), or imports an estimator package fails the
  `layering` gate in `tools/compliance.py`. Solvers take arrays and a frozen config
  dataclass and return a frozen state dataclass, so the numerics can be tested without
  constructing an estimator. SciPy and threadpoolctl, scikit-learn's own dependencies,
  are imported only inside the functions that use them; a switch a solver honors, such
  as `LogisticRegression`'s dense preconditioner (`Tuning.lbfgs_dense_preconditioner`),
  is a field of its config, which the estimator fills from `get_tuning()`. `_kernels` may
  read `get_tuning()` for block sizes, which are private typed knobs, not user
  configuration. The gate is static, so an indirect read — `inference_device(backend)` or
  `use_device(None)` resolving the configured device — is still on the reviewer. Neither
  layer makes one: solvers pass their config's device explicitly, and an estimator
  resolves its device with `inference_device()` and passes it to the kernels and the
  neighbor index as `device=`.
- A `sklearn._*` import outside `patching/` fails CI. Inside `patching/` it must be on an
  explicit allowlist in `tools/compliance.py`, which is currently empty.
- Layer 1 raises on a capability mismatch. Layer 2 falls back. The whole difference is
  `_mlxlearn_allow_fallback`, and any behavior that cannot be expressed as flipping that
  flag is a design error.

## Adding an estimator

**Subclass the scikit-learn estimator.** Never redefine `__init__`. Subclassing inherits
the exact constructor signature, `get_params`/`set_params`, `_parameter_constraints`, and
the estimator tags, so `clone`, `Pipeline`, `GridSearchCV` and `isinstance` keep working —
and `super().fit(...)` becomes an exact scikit-learn fallback with no extra machinery.

The shape of a `fit`:

```python
class MLXThingMixin(BackendMixin):
    # Fitted attributes only the MLX path creates. _clear_backend deletes them, so a
    # refit that ends on scikit-learn (or fails) does not keep the old device state
    # alive in memory and in pickles.
    _mlxlearn_mlx_attrs = ("_device_state_",)


def fit(self, X, y):
    self._validate_params()   # scikit-learn's constraints: overriding fit bypasses its decorator
    self._clear_backend()     # forget the last fit; the estimator is unfitted until _set_backend
    backend = self._select_backend(
        "fit",
        shape=_safe_shape(X),                          # _common.validation: the raw shape
        crossover=self._below_crossover,               # (n_samples, n_features) -> (below, detail)
        capability_check=lambda: self._check_params(X),   # parameters and sparsity: cheap
        data_check=lambda: check_float32_range(X),        # O(n·d) scans: skipped on a handoff
    )
    if backend == "sklearn":                           # only reachable from Layer 2
        super().fit(X, y)
        self._set_backend("sklearn")
        return self
    X, y = validate_data(self, X, y, ...)
    with mlx_guard(type(self).__name__, "fit"):
        state = solve(X, y, ...)                       # locals only, evaluated before return
    self.coef_ = state.coef                            # publish
    self._set_backend(backend)                         # the commit point: fitted from here
    return self
```

`_select_backend` decides in a fixed order: the capability check, then
`MLXLEARN_FORCE_BACKEND`, then size, then the data checks, then the device. The order is
the point. "We cannot do this at all" outranks "this is small", so a capability mismatch
is always reported as one. And when patched mode is about to hand a fit to scikit-learn
by size, or by a rule of its own passed as `handoff=` (the neighbor estimators'
narrow-data rule), the decision comes from `X.shape` alone, before any validation, scan or
copy — scikit-learn validates the data itself, and scanning it first made patched fits slower
than the stock ones they replaced. Input with no shape before validation (a list)
gets a provisional `"cpu"`, with nothing recorded and no device resolved, and the fit
decides again once validation has given it a shape. The three 0.1.0 estimators all
decide through these keywords. `SVC`'s data-dependent limits need the validated array
and labels, so it checks them after its second decision rather than in `data_check`, and
routes a refusal through `_handle_capability_error` by hand.

Each decision is recorded when it is made, and a capability fallback warns there, before
scikit-learn does any of the work, so an `"error"` warnings filter fails fast.
`_select_backend` may run more than once in one fit: a later decision takes the earlier
one's place in the record, and so does what the work found (`_amend_fit_event`, how a
solver stopped), so one fit is still one event. `_set_backend(backend)`, called exactly
once at the end of every successful fit, is the commit point: it sets
`execution_backend_` and keeps the fit's event on the estimator as `fit_event_`. A fit
that raises after deciding leaves its decision on the record, once, and the estimator
unfitted; inference then raises `NotFittedError`.

And the shape of every inference method:

```python
def predict(self, X):
    check_is_fitted(self)
    if self._fitted_backend("predict") == "sklearn":
        self._note_inference("predict")
        return super().predict(X)
    ...                                                # _note_inference where the work is routed
```

`BackendMixin` wraps every public inference method a subclass defines or inherits as a
plain function (`predict`, `predict_proba`, `decision_function`, `kneighbors`, and
scikit-learn's own `score`, `kneighbors_graph`, `radius_neighbors_graph`, …), so one
public call records one event, named after the method the caller invoked, however many of
the others it reaches internally. `available_if` descriptors are left alone, because
wrapping one would make the method exist on estimators that do not have it; a class whose
`available_if` method must record its call redeclares it as
`available_if(check)(_public_call(name, fn))`, as `SVC` does for `predict_proba` and
`predict_log_proba`. An inference path that records something other than a sticky event —
a fallback at inference — claims the call's one event with `_claim_call_event`.

Inference never re-decides the backend. That is the sticky invariant, and it is what stops
an MLX-fitted model from wandering into scikit-learn's inference path holding state that
path cannot read. There is one carve-out, and it is Layer 2's: a sparse query to
`kneighbors`, `kneighbors_graph`, `predict` or `predict_proba` of a neighbor estimator
fitted on MLX is answered by scikit-learn's brute-force search, which reads the fitted
state the MLX path populates for it (`_fit_X`, `_fit_method="brute"`,
`effective_metric_`), and is recorded as a fallback. The query is cast to `_fit_X`'s
float32 first: scikit-learn's search needs the two to share a dtype, and a float64 query
fell to a slower pairwise path that cannot compute `"sqeuclidean"` on sparse input. A
GPU-fitted model running on MLX's CPU in a process without a GPU (`inference_device`) is
not a carve-out: it is the same code on another MLX device, recorded as
`"gpu-unavailable"`. Inference that is host code on the fitted state
(`LogisticRegression`'s, `NearestNeighbors.radius_neighbors` and
`radius_neighbors_graph`) uses no device: `_note_inference(..., host=True)` records it as
`"sticky"` and resolves no device, so `device="gpu"` has nothing to refuse. The radius
methods take a sparse query that way too, in both layers — scikit-learn's code serves
it — so it is an ordinary sticky call, not a fallback, and `fallback_policy` does not
apply to it.

Wrap every MLX block in `mlx_guard(...)` so an unexpected failure becomes
`BackendExecutionError` rather than a silent fallback.

### The gates, per estimator, before merge

1. scikit-learn's estimator checks — `check_estimator(on_fail=None)` over both layers, at
   the default configuration and with the MLX GPU path forced, plus
   `check_param_validation` — in `tests/test_sklearn_compliance.py`. Any xfail is
   justified under a named rule in [`fp32_policy.md`](fp32_policy.md) §4 and listed in
   `EXPECTED_XFAILS`.
2. `clone`, `get_params`/`set_params` round-trip, `Pipeline`, `GridSearchCV`,
   cross-validation, `pickle`, `joblib`, refit, `n_features_in_`. `tests/unit/test_gates.py`
   runs `joblib` (memory-mapped too), process-parallel cross-validation and a `Pipeline`
   inside `GridSearchCV` for every registered estimator in both layers; the check suite
   covers clone, parameters, refit and `n_features_in_`, and
   `tests/unit/test_invariants.py` pickling and fit purity.
3. Numerical parity against stock scikit-learn at the tolerances in `fp32_policy.md` §3 —
   including the objective value, not only the coefficients. Two correct solvers land in
   different places on an ill-conditioned problem, so a coefficient-only test fails for a
   correct implementation and passes for a subtly wrong one. `tests/parity/`.
4. Precise errors on the Layer 1 path; recorded fallbacks on the Layer 2 path — exactly one
   fit event per fit, matching `execution_backend_` and `fit_event_`, and one event per
   public inference call, named after it (`tests/unit/test_runtime.py`, and
   `tests/unit/test_event_invariants.py` over every fit path of every estimator in both
   layers).
5. Sticky-backend tests: inference reuses whatever `fit` chose
   (`tests/unit/test_gates.py`; `tests/unit/test_event_invariants.py` for the event each
   inference call records, GPU-less hosts included).
6. A benchmark entry in `benchmarks/run.py` with a measured crossover wired into
   `_common/config.py`'s `Tuning`.

An estimator is registered in `patching/_registry.py` only after all six pass. Registering
early means silently swapping something untested into someone's `sklearn` namespace.

## Things that are not allowed

**No module-level mutable state, and no cache keyed on data.** The ancestor kept weakrefs
to the last training and query matrices so it could skip re-uploading them, plus `id()`-keyed
LRU caches in the estimator layer. CPython recycles `id()` after a garbage collection, so a
freed array whose address was reused produced predictions computed on the wrong data,
silently. Everything cached lives on the fitted estimator and dies with it. What a process
may keep is a description of itself, keyed on nothing a fit passes in: the compiled Metal
kernels (below), the host thread pool (`_common/host.py`), and the BLAS pools that
`LogisticRegression`'s dense-preconditioner plan holds to one thread while it builds its
model (`_solvers/logistic.py`), with the lock that stops two plans on two threads from
restoring the thread count out of order. That limit is process-wide, and the lock orders
mlxlearn's plans only: another thread's threadpoolctl limit that begins inside one and
ends after it restores the plan's single thread when it ends. So the plan takes it only
around the model, which only a fit the gate admits builds, and not at all where every
pool already runs one thread or where there is no pool to limit (Accelerate).

**No aliasing of a caller's array.** `fit` copies. An estimator that holds a view of the
caller's buffer changes its own predictions when the caller mutates their array.

**No broad `except Exception` around accelerated work.** The ancestor's dispatcher caught
everything and reran on scikit-learn, so a `ValueError: Converting -1 to uint32 would result
in overflow` from a padding bug reached users as a slightly slow but correct answer. That
bug survived for exactly as long as the `except` did.

**No plain `ValueError` or `TypeError` for bad input inside `mlx_guard`.** The guard lets
only the `MLXLearnError` family, `MemoryError` and escalated warnings through; everything
else — including the `ValueError`s and `TypeError`s MLX raises for its own shape and
argument errors — becomes `BackendExecutionError`. A deliberate input error detected
inside a guarded block is therefore an `InvalidInputError`, which is a `ValueError`, so
scikit-learn-facing behavior is unchanged. A capability limit discovered there is a
`CapabilityError` as usual, but `_select_backend` has already returned, so nothing turns
it into a fallback: find it before, or route it through `_handle_capability_error` by
hand, as `SVC` does for its data-dependent limits and the neighbor estimators for a
training span their index cannot hold.

**No `patch_sklearn()` advice in a capability message.** A `CapabilityError`'s message says
what cannot be served; the remedy for a directly imported estimator ("run under
`patch_sklearn()`", "compute these with scikit-learn") goes in `advice=`. Layer 1 raises
both; a patched estimator, already under `patch_sklearn()`, warns and records `.summary`
followed by what it did. An error at inference, where neither layer can fall back, advises
no fallback.

**No unevaluated `mx.array` on shared state.** MLX is lazy: an `mx.array` is a graph until
something evaluates it. One published on an estimator or an index that other threads read
— a device-resident training copy, a prepared operand — gets evaluated by whichever thread
touches it first, concurrently with the others. Concurrent first queries on an unpickled
neighbors model could segfault while two threads built the same device state, and the
rule has since caught a lazy transpose published on an index. Build device state into
locals, `mx.eval` it, and publish it with one assignment, under a lock if it is built
lazily, so a reader sees either nothing or the finished state.

**No data scan before a by-design handoff.** A patched fit headed for scikit-learn on size
alone must cost what scikit-learn costs. Decide from the shape, and keep the O(n·d)
capability checks — float32 range, target type, weight signs — in `data_check`, which
does not run on that path.

**No approximation under a scikit-learn estimator name.** If an estimator cannot implement
the objective its name promises, it does not ship under that name.

**No silently ignored parameters.** The ancestor accepted `metric_params` and computed
unweighted Euclidean distances anyway. A wrong answer delivered quickly is the worst
outcome available.

**No `mx.random.seed`.** It is process-global and belongs to the application, not to a
library it imported. Use explicit keys through `_common/rng.py`. The `layering` gate rejects
any `*.random.seed` — MLX's, NumPy's or the standard library's — anywhere in the package.

## mlxlearn's own Metal kernels

`mx.fast.metal_kernel` compiles a Metal source string at run time, so the wheel stays
pure Python. Two kernels ship: the neighbor search's fused tile and selection
(`_kernels/_topk.py`), and SVC's one-vs-one SMO, every pair of a multiclass rbf fit in one
launch (`_kernels/_ovo.py`). Every kernel goes through `_kernels/_metal.py`, which holds
these rules; each was measured on MLX 0.31.2 and 0.32.2 before the first kernel shipped.

- **The gate is the resolved device, and MLX 0.32.** `_metal.kernels_enabled(device)` takes
  the `"gpu"` or `"cpu"` the caller resolved and passes its built-in path — never MLX's
  default device: on 0.32 a kernel launched without `stream=` ran on the GPU inside
  `use_device("cpu")`, silently. Every launch names `mx.gpu` (`_metal.launch`). MLX 0.31
  keeps compiled libraries in an unlocked map keyed by name alone — one kernel fed inputs
  of two sizes from two threads crashed the process — so the gate is closed there, and
  MLX's CPU device and 0.31 run MLX's own operations, as they always did. So does
  `MLXLEARN_METAL_KERNELS=0` (`Tuning.metal_kernels`), for tests and bisection.
- **A kernel that must match MLX bit for bit lists the MLX releases it was checked on.**
  The fused selection reproduces MLX's `argpartition` and matmul bits through behaviors
  MLX does not promise (`docs/fp32_policy.md` §2.1), so `_topk.VALIDATED_MLX` names the
  releases, to the patch (`(0, 32, 2)`), on which every identity test passed, and any
  other — a later patch release of the same minor one included, since which GEMM kernel
  MLX picks for a shape can change in one — keeps MLX's own selection until someone runs
  them there and adds it. A run-time self-test does not replace the list: the behaviors
  depend on the shapes, which a test small enough for a first query does not reach.
  **A kernel whose every answer the host verifies needs only the gate.** The one-vs-one
  SMO keeps its state in double-float, which no MLX operation reproduces, so nothing it
  returns is trusted: every pair is held to the host solver's float64 stop and finished
  there if it fails (`smo.finish_c_svc`, `docs/fp32_policy.md` §2.5). A release that
  changed its behavior could cost time, never an answer, so it runs on any MLX the
  gate admits.
- **`#pragma METAL fp contract(off)` first** (`_metal.PRELUDE`, prepended by
  `_metal.metal_kernel`). Metal's default mode fuses `a*b + c` written as one expression
  into an fma, which rounds once where MLX rounds twice; `math_mode` is pinned to
  `"safe"`, since the relaxed and fast modes fuse across statements too. Spell out
  `metal::precise::` for `exp`, `log`, `sqrt` and `rsqrt`: on 0.32 the bare names are the
  fast approximations. Use no float atomics, merge in a fixed order, and call
  `simd_min`/`simd_max` on 32-bit values only — Apple GPUs reduce a 64-bit one through
  float32. Reduce with maxima where you can: a maximum with ties broken by index is the
  same in any order, which is what makes the one-vs-one kernel's pairs independent of
  their threadgroup width and launch. `fma` appears only where a kernel means it — the
  exact product error of a double-float multiply.
- **Inputs of at least eight elements** (`_metal.pad_min8`); sizes travel in one int32
  array of eight (`_metal.int_params`), never as template values, and a value int32
  cannot hold is refused, in a launch table of offsets too (`_ovo.waves` keeps a wave's
  packed Grams under 2³¹ entries). MLX declares a shorter input `constant`, which
  compiles a second variant of the source. Sources are ASCII and names fixed: the
  operating system's Metal shader cache serves a library to later processes only while
  both stay the same, and a source change bumps the name's version.
- **Kernel objects come from an `lru_cache` factory keyed by static parameters only** —
  the variant, never data — and kernels stay out of `mx.compile`d functions.
- **Compile on first use, in the query that needs it, not ahead of it.** MLX holds a lock
  that stalls every other GPU call in the process while it builds a library, so a
  background build would stall the user's GPU work to save a first fit tens of
  milliseconds, once per machine — and one started at the index build overlapped only
  that build's host work, 8 ms of it at 30 000 × 256. A first use runs the kernel on a
  fixed input whose answer the host computes, under a lock, so concurrent first queries
  compile once; a kernel that does not build (MLX raises `RuntimeError`), whose widest
  threadgroup the compiled pipeline cannot take (`ValueError`), or that answers wrongly
  is not used in that process. That narrow `except` is a capability probe on fixed data,
  like `gpu_available()`, never a guard around a query's own work.
- **`mx.eval` a launch before NumPy reads it.** MLX raises those errors while it
  evaluates. From `mx.eval` they are Python exceptions — the first-use check's `False`, a
  fit's `BackendExecutionError` — but on MLX 0.32.2 an evaluation that runs inside
  NumPy's buffer protocol (`np.array`, `np.asarray`) turns the same error into
  `std::terminate`, and the process aborts. The tests make the first launch fail both
  ways, and a fit's own launch fail after the check passed.
- **Launches stay short and bounded.** A kernel that iterates (the one-vs-one SMO) runs
  a step budget per launch — 2 048 steps first, then about a quarter second's worth by
  the last launch's time per step — and the host relaunches with the state on the device,
  so no command buffer nears the GPU watchdog; every launch advances or stops what it
  runs, a pair it neither advanced nor stopped is the host's to finish, and a hard
  launch count backs both.
- **Device memory has a budget the documentation already promises.** The one-vs-one
  kernel holds every pair's Gram it solves at once, so it solves them in **waves** whose
  packed Grams fit `cache_size` (the kernel cache a fit already sizes, within the
  `Tuning.smo_kernel_cache_*` ceiling), and gives every wave of a fit one buffer size, so
  MLX's buffer cache reuses it. Everything else it puts on the device to build them —
  the training rows it keeps there, which stay only while they take at most half the
  budget, and every transient, the rows' squares included — fits
  `Tuning.distance_block_bytes`; a pair too large for that alone takes what
  `kernel_matrix` takes to build it on the host path. Count every array a lazy graph
  materializes, not only the tiles: an uncounted copy of the rows and their squares put
  a 20 000 × 4 096 fit at over 1 GiB against a 456 MiB budget. The buffers are locals,
  freed when the fit returns, and the existing cache-release rule (`release_mlx_cache`)
  applies on the way out.
- **Tests prove which path ran**, by spying on the kernel's launches, and compare its
  answers with the switch off — bit for bit where the kernel promises identity: indices,
  distance bits and diagnostics counts (`tests/unit/test_fused_selection.py`); to the
  host solver and to scikit-learn at the `docs/fp32_policy.md` §3 tolerances where it
  promises an optimum, with every gate miss bit for bit the host loop
  (`tests/unit/test_ovo_kernel.py`, `tests/parity/test_svc_ovo.py`). On MLX 0.31, the
  minimum-versions job's, tests check that no kernel object is ever built.

## Compliance

```bash
python tools/compliance.py                     # all gates
python tools/compliance.py licenses --json     # some gates; stdout is pure JSON
```

Six mechanical gates — branding, private imports, upstream headers, dependency licenses,
layout, and layering — run on every push. The license gate walks the whole runtime
dependency closure as installed, not only the direct dependencies, and parses SPDX
expressions rather than matching substrings; a package whose metadata cannot be read
mechanically needs a reviewed entry in `LICENSE_OVERRIDES`, with a reason.

## Testing

- **GPU.** Tests that need a working Metal device either carry `@pytest.mark.gpu` or reach
  the GPU implicitly and fail with `DeviceError` without one. On a machine with no GPU,
  both are skipped with a reason rather than failing by the dozen. The implicit skip only
  catches a `DeviceError` that reaches pytest unwrapped — not one inside `GridSearchCV`'s
  "all fits failed" — and without a GPU most MLX work raises nothing: auto-selection picks
  the CPU, and a forced MLX fit runs on MLX's CPU device. So a test that asserts the MLX
  path ran must carry the marker. The slow test `tests/tooling/test_gpu_policy.py` runs the
  default suite on a simulated GPU-less host to hold every test to that. CI sets
  `MLXLEARN_REQUIRE_GPU=1`, under which a missing GPU stops the session instead, so a
  runner that loses Metal cannot skip its way to green. The policy is `tests/_gpu.py`.
- **pandas** is a test dependency (`pandas>=2.2.3` in the `test` and `dev` extras, the floor
  installed exactly by the minimum-versions job), so the tests of how mlxlearn reads data
  frames run on every CI leg. torch is not: the torch container cases in
  `tests/unit/test_validation.py` run only where it is installed — the nightly job, and
  local runs.
- **Doctests** run as their own CI step: `pytest --doctest-modules src/mlxlearn`.
- **`tests/tooling/`** tests the repository's tooling — this file's gates, the benchmark
  harness and gate, the workflows' pins — and is left out of the sdist with the tooling.
- **`tools/check_environment.py`** prints the versions and device a CI job is testing and
  fails if they are not what the job claims: unmet requirements, a stale install whose
  metadata disagrees with `mlxlearn.__version__`, or (`--constraints`) a pinned floor that
  did not install exactly.

## Benchmarks

```bash
python -m benchmarks.run --profile smoke
python -m benchmarks.run --profile full --output bench/results.json --report bench/report.md
python -m benchmarks.run --profile published --report bench/published.md
python -m benchmarks.gate --results bench/results.json
```

`smoke` is small enough for a laptop and still lands on both sides of every `Tuning`
threshold (`tests/tooling` checks that it does); `full` is the regression grid;
`published` holds the cases `docs/benchmarks.md` reports, with the settings they were
measured under (Layer 1, forced onto MLX).

The thresholds themselves come from `benchmarks/calibrate.py`, one named grid per rule:

```bash
python -m benchmarks.calibrate run knn --output bench/calibration/knn-1.9.0.jsonl
# ...again under the other end of the supported scikit-learn range, then:
python -m benchmarks.calibrate update docs/benchmarks.md bench/calibration/*.jsonl
```

A change to a `Tuning` threshold, or to the rule that reads it, comes with the grid that
justifies it, run under both scikit-learn versions, its raw results in
`benchmarks/results/calibration/`, and the table in `docs/benchmarks.md` re-rendered from
them; `tests/tooling` fails while that table is stale, and checks that every threshold's
grid straddles it. `benchmarks/ab.py` times one source tree against another in fresh
processes, for the CHANGELOG's release-over-release numbers.

Each result records the backend the mlxlearn side ran on, and the gate judges those
decisions — the configured `Tuning` — rather than a crossover inferred from the run it is
judging: a problem `Tuning` sends to MLX that loses, and a patched handoff whose own
overhead loses, both fail. A backend forced with `--force-backend` or with
`MLXLEARN_FORCE_BACKEND` is recorded as forced, the same way, so the gate does not charge
it to `Tuning`; the environment record includes the effective `Tuning` and every
`MLXLEARN_*` variable. A difference counts only when the medians are three standard
errors apart, each estimated from a real interquartile range. That bounds the false-alarm
rate per case; a run gates over a hundred cases, so the runner re-measures any case the
gate would fail over three times the iterations, records it as the result's
`confirmation`, and the gate fails the case only if the loss reproduces there. A case's
own iteration count caps `--iterations`; it never raises it.

Shared CI runners do correctness only: CI runs the smoke profile with one iteration, which
checks the harness, the gate, and the result schema, and can never be significant.
Absolute timings come from the nightly job on fixed self-hosted Apple hardware, because a
timing gate on a shared, throttled, heterogeneous runner measures the runner. Its schedule
waits for the repository variable `MLXLEARN_BENCH_RUNNER` to be `true`, which says the
`mlxlearn-bench` runner exists; a manual dispatch always runs.

Everything is `mx.eval`-synchronized — MLX is lazy, so an unsynchronized benchmark times
how long it took to *schedule* the work — warmed up, reported as a median with its spread,
and compared against scikit-learn in interleaved pairs so a thermal ramp degrades both
sides equally rather than whichever ran second. Within a pair the two sides take turns
going first, and every timed call gets the same preparation, whichever backend it runs
on: a garbage collection before it and a synchronization inside its window.

## Releasing

1. Every gate green; `python -m benchmarks.run --profile full` re-run and
   `docs/benchmarks.md` refreshed.
2. `CHANGELOG.md` updated.
3. Version bumped in `src/mlxlearn/__init__.py` — the only place it lives. `pyproject.toml`
   reads it (`dynamic = ["version"]`), statically, so a build never imports MLX. After a
   bump, reinstall (`pip install -e .`): an editable install's metadata is frozen at
   install time and otherwise keeps reporting the old version.
4. README links stay absolute (`https://github.com/tabulai/mlx-learn/...`): the README is
   also the PyPI page, where a relative link is a 404. `tests/tooling/` checks this.
5. Tag `vX.Y.Z`. The release workflow runs the whole CI workflow on the tagged commit and
   does not build anything itself: CI's build job builds the `dist` artifact and checks
   it — a `py3-none-any` wheel that installs into a clean environment with metadata that
   agrees with `mlxlearn.__version__`, an sdist that carries the test suite, and that
   suite collecting against the installed wheel. The release workflow's `version` job
   then requires the wheel's filename, its metadata, the sdist's filename and the tag to
   agree. Only then does it publish exactly those files, through PyPI Trusted Publishing
   from a `pypi` environment that requires manual approval. No API token is stored in this
   repository. Publishing takes both a `v*` tag push and that approval; a manual run of
   the workflow is a dry run that never publishes.
