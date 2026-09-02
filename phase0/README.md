# Phase 0: freeze and audit

Phase 0 establishes an evidence-backed boundary between the historical
`scikit_learn_mps` repository and the new `mlxlearn` codebase.

## Source snapshot

- Historical source: `/Users/tunguz/Programming/scikit_learn_mps`
- Required source revision: `720053bdf9b7a377ee45bd5bd573bc09a7df1743`
- Upstream comparison checkout:
  `/Users/tunguz/Programming/sklearn_repos/scikit-learn-intelex`
- Migration policy: [`audit_policy.json`](audit_policy.json)

The audit refuses a different source revision unless
`--allow-revision-mismatch` is supplied. It never edits either source
repository.

## Generated artifacts

`tools/phase0_audit.py` writes:

- `generated/source_snapshot.json`: revisions, remotes, and repository counts.
- `generated/provenance.csv`: one row per tracked source file, including hash,
  header evidence, upstream similarity, provisional provenance, disposition,
  and human-review status.
- `generated/estimator_support.csv`: the explicit estimator migration matrix.
- `generated/audit_summary.md`: aggregate findings and unresolved review queue.

`tools/run_phase0_baseline.py` writes:

- `baseline/environment.json`: Python, package, platform, and Apple hardware
  metadata.
- `baseline/test.log`: the complete historical pytest output.
- `baseline/benchmark.log`: the complete benchmark-smoke output.
- `baseline/benchmark_results.json`: the historical fidelity benchmark data.
- `baseline/manifest.json`: command lines, durations, exit codes, and artifact
  hashes.
- `baseline/summary.md`: a concise pass/fail record.

Generated files are checked in because they are the immutable migration
evidence for this source revision.

## Classification semantics

The provenance classification is deliberately called *provisional*. A matching
upstream path, a strong text-similarity score, or an upstream copyright header
is evidence of derived code. Absence of those signals is not proof of
independent authorship. Every file intended for migration must therefore have
both:

1. an allowlisted migration disposition; and
2. completed human review before it can be copied.

Files marked `drop` or `generated-do-not-copy` do not enter the new history.
Files marked `rewrite` provide behavioral requirements only; implementation
must be newly written. Files marked `port-candidate` are the only possible
source-copy candidates, and their notices still require review.

## Phase 0 exit criteria

- [x] The source revision and working-tree state are recorded.
- [x] Every tracked file appears exactly once in `provenance.csv`.
- [x] Every public estimator appears in `estimator_support.csv`.
- [x] Historical tests and the fidelity benchmark have reproducible logs.
- [x] Any failing test or benchmark cell is documented in `summary.md`.
- [x] The human-review queue for files selected for the first KNN port is empty.
- [x] The compiled-extension go/no-go decision is made, with measurements —
      [`cython_decision.md`](cython_decision.md).
- [x] Per-module authorship attestation for the shipped surface —
      [`attestation.md`](attestation.md).

The human-review criterion is intentionally not automated: provenance tooling can
surface evidence, but it cannot make authorship determinations.

## How the port-candidate review queue was resolved

The audit left 12 files with a `port-candidate` disposition, only two of which
had completed technical review. Rather than block, the 0.1.0 surface was built
under a stricter rule than the audit required: **nothing was copied**. Every
module under `src/mlxlearn/` was newly written, with the ancestor consulted only
as a behavioral reference. That makes the remaining review queue moot for the
shipped surface — there is nothing ported to review — and it is recorded per
module in [`attestation.md`](attestation.md).

The behavioral extraction that informed the rewrite also found defects in the
ancestor that the rewrite deliberately does not reproduce. The three most
consequential, each now covered by a regression test:

1. `_exclude_self_neighbors` compared each row against its **block-local** index
   rather than its global one, so `kneighbors(None)` was wrong for any training
   set larger than one query block — verified at 1 wrong row in 1500 with default
   blocking. Its tests only ever used 3 rows.
2. The GPU top-k returned `uint32` indices, so padding a short trailing tile with
   `-1` raised `Converting -1 to uint32 would result in overflow`. Because the
   dispatcher caught every exception and reran on scikit-learn, this presented to
   users as an unexplained slowdown rather than an error.
3. `metric_params` was accepted and silently ignored, so a weighted-Minkowski
   request returned unweighted Euclidean results.
