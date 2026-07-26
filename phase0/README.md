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

- The source revision and working-tree state are recorded.
- Every tracked file appears exactly once in `provenance.csv`.
- Every public estimator appears in `estimator_support.csv`.
- Historical tests and the fidelity benchmark have reproducible logs.
- Any failing test or benchmark cell is documented in `summary.md`.
- The human-review queue for files selected for the first KNN port is empty.

The last criterion is intentionally not automated: provenance tooling can
surface evidence, but it cannot make legal authorship determinations.
