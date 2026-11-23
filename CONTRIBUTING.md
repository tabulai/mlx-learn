# Contributing to mlxlearn

This file describes how work actually lands in this repository. It does not introduce
policy. The rules it points at live in [`docs/development.md`](docs/development.md) —
layering, the estimator gates, and the list of things that are not allowed — and that
file remains the authority. What follows is the part that was previously learnable only
by reading merged pull requests.

The short version: **a change to numerics is not finished when it is correct. It is
finished when it is measured, and the measurement is committed.**

## Setting up

```bash
git clone https://github.com/tabulai/mlx-learn
cd mlx-learn
pip install -e ".[dev]"
```

Then establish a baseline *before* you change anything:

```bash
pytest
ruff check src tests tools benchmarks
python tools/compliance.py
```

A clean run has skips. Tests that need a GPU or an optional native extension skip by
capability rather than fail, so "N skipped" is the expected state, not a problem to fix.
If `pytest` reports failures on an untouched checkout, they are almost certainly
environmental — record them now, because you will want to prove later that they were not
yours.

mlxlearn requires macOS on Apple silicon. The package imports elsewhere and its CPU path
runs, but nothing there is tested or supported.

## The two layers

Layer 1 is the library: explicit imports, strict errors, no mutation of scikit-learn.
Layer 2 is the patching shell, the only package allowed near scikit-learn's namespace.
The split is enforced by `tools/compliance.py`, not merely described.

Read [`docs/development.md`](docs/development.md) before writing code. In particular, the
"Things that are not allowed" section is written as a list of bugs that actually shipped
in a predecessor project, each with the failure it caused. A pull request that
reintroduces one reads as not having opened the file.

## The gates, and which ones you can close

[`docs/development.md`](docs/development.md) lists six gates for a new estimator. It does
not say where each runs, which matters a great deal when you are planning work.

Gates 1 through 5 run in CI on shared `macos-15` runners, across a paired — not
cross-product — matrix of Python and scikit-learn versions:

| Python | scikit-learn |
|---|---|
| 3.10 | 1.7.2 |
| 3.11 | 1.8.0 |
| 3.12 | 1.9.0 |
| 3.13 | 1.8.0 |
| 3.13 | 1.9.0 |

Version-dependent scikit-learn behaviour is a live failure mode here, not a theoretical
one. Deprecations in this range arrive as string sentinel defaults, and
`docs/support_matrix.md` lists the three that currently bite.

**Gate 6 you cannot close.** It requires a measured crossover wired into `Tuning`, and
absolute timings come from the nightly job on fixed self-hosted Apple hardware.
`.github/workflows/nightly-benchmarks.yml` triggers only on `schedule` and
`workflow_dispatch` — there is no `pull_request` trigger, so no contributor pull request
ever produces an official timing. Supply your own numbers as supporting evidence, name
the hardware you ran on, and say plainly in the pull request that gate 6 needs a
maintainer run.

If this is your first contribution, your workflow runs need maintainer approval on every
push. Expect CI to sit idle until someone approves it.

## Measurement is the deliverable

Every merged pull request that changes numerics ships four things: the source diff, a
disproportionately larger test diff, a committed benchmark driver with its raw JSON
results, and a dated report under `docs/`. There is no counterexample in the history.

### The dated report

Named `docs/<topic>-<YYYY-MM-DD>.md`. Six parts, in this order:

1. **Headline.** The before and after in the opening paragraph, with the exact workload
   attached — not "faster", but the two numbers and the shape they were measured on.
2. **What changed.** The mechanism in prose, including the inequality or bound the change
   exploits.
3. **Results tables.** With the baseline *and* candidate commit SHAs printed on both
   sides, so a reader can reproduce the exact pair.
4. **Correctness as a count.** Not an adjective. "Across all 1,116 warmup and timed
   queries, the two revisions returned bit-identical distance buffers" is the form. Give
   the full-suite result with exact pass and skip counts, and a reason for each skip.
5. **Measurement.** Chip, RAM, macOS version, Python version, MLX version, NumPy version,
   warmup count, number of paired rounds, whether execution order was alternated, and
   what is excluded from the timed interval.
6. **Reproduction.** A copy-pasteable command carrying the exact SHAs, with links to the
   driver and the raw JSON.

[`docs/knn-tight-budgets-2026-11-17.md`](docs/knn-tight-budgets-2026-11-17.md) is the
cleanest worked example. Read one in full before writing your own.

Everything is `mx.eval`-synchronized. MLX is lazy, so an unsynchronized benchmark times
how long it took to *schedule* the work. Warm up, report medians with their spread, and
compare against scikit-learn in interleaved pairs so a thermal ramp degrades both sides
equally rather than whichever ran second.

### Report your own bad news

This is the convention most likely to be missed, because every instinct runs the other
way. The reports in this repository volunteer their failures in the same tables as their
wins:

- Regressions are named with numbers, in the headline table, not in a footnote.
- Unstable measurements are reported rather than re-rolled until they look better.
- Non-determinism in the *baseline* is disclosed.
- Claims the data does not support are explicitly withdrawn, in writing.
- Seeds, thermal state and sample counts are always stated.

[`docs/radius-neighbors-2026-11-22.md`](docs/radius-neighbors-2026-11-22.md) carries both
a "Remaining losses" section and a "Limitations and follow-ups" section, listing every
cell the change still loses and why. A report with no losses in it invites the question
of what was not measured.

## Pull requests

Bodies run from a short paragraph to about a page; the median is a few hundred words. The
headings settle into Summary, Evidence or Results, and a validation section.

Two conventions you would not guess:

- **"Alternatives I rejected"** — a list with one line of reasoning each. A rejected
  option with a stated reason is worth more to a reviewer than the chosen one with none.
- **A Test plan whose unchecked boxes are left unchecked.** `- [ ] Not run on M1 Ultra`
  is a disclosure, not an oversight. The convention is to enumerate what you did *not*
  verify, and leave the box empty so it is visible.

There are no formal GitHub reviews on any pull request in this repository. Feedback
happens out of band, which means a pull request that stalls stalls silently. If yours has
been quiet, the issue thread is a better place to push than the pull request.

Open an issue before a large change. The two outside contributions this project has
received were each measured against a configuration grid in the hundreds, drew dozens of
confirmed review findings, and were substantially rewritten before merging. Arriving with
the grid and the capability contract already settled is what shortens that.

## Commits and branches

Subjects are imperative and run roughly 50 to 80 characters, stating the effect rather
than the mechanism — "Certify Lloyd assignments from current center separation", not
"Change the assignment loop". There are no Conventional Commits prefixes anywhere; do not
introduce them. About half of all commits carry a body, of one to three short paragraphs.

Branches are `codex/<topic>`, `claude/<topic>` or `feat/<topic>`. Merges are real merge
commits, never squashes, so your individual commits survive into history and are worth
writing carefully.

Long-lived branches are kept current by merging `main` in, with a comment on the pull
request describing exactly which conflicts were resolved and what did not change.

## Traps that fail CI

Each of these is obvious afterwards and invisible beforehand.

- **Adding a benchmark case breaks a test.** `tests/unit/test_grid_refresh_benchmark.py`
  asserts an exact case count against `benchmarks.run.PROFILES["full"]`, plus exact
  per-algorithm counts. Update it in the same commit.
- **The private-import allowlist is empty.** `ALLOWED_PRIVATE_SKLEARN_IMPORTS` in
  `tools/compliance.py` is `{}`. Any `sklearn._*` import outside `patching/` is rejected,
  and inside `patching/` it must be added to that allowlist with a reason.
- **The branding gate scans documentation too.** Naming the predecessor project or its
  vendor in a new document fails CI unless the file is exempt.
- **`ruff` runs on `benchmarks/`.** A throwaway driver still has to lint.
- **The xfail list is empty by policy.** `EXPECTED_XFAILS` in
  `tests/test_sklearn_compliance.py` is `{}`, and `docs/fp32_policy.md` §4 names checks
  that are never xfailed under any circumstances. If your estimator needs an xfail, it
  needs a named rule in that file first.
- **A new module containing `except Exception` fails a test.** `tests/unit/test_invariants.py`
  scans every module for broad exception handlers against an explicit allowlist. Wrap MLX
  work in `mlx_guard(...)` instead.

## Adding an estimator

[`docs/development.md`](docs/development.md) has the shape of a `fit`, the subclassing
rule, and the gate list. Beyond that, registration touches a fixed set of places:
the estimator module and its package `__init__`, a solver under `_solvers/` taking a
frozen config and returning a frozen state, `patching/_estimators.py` and
`patching/_registry.py`, the Layer 1 and Layer 2 lists in `tests/test_sklearn_compliance.py`,
the invariants list in `tests/unit/test_invariants.py`, the patched-name assertions in
`tests/patching/test_patch_sklearn.py`, a new parity test, and then `CHANGELOG.md`,
`README.md` and `docs/support_matrix.md`.

Register last. An estimator earns its `patching/_registry.py` entry only after all six
gates pass; registering early silently swaps something untested into a user's `sklearn`
namespace.

## Writing style

The prose here is declarative and gives the reason for every refusal. The characteristic
move is to name the bug that motivated a rule rather than assert the rule abstractly — see
`docs/development.md` on broad exception handlers, or the module docstring in
`src/mlxlearn/svm/_classes.py` on why no approximation ships under the `SVC` name.

Two habits worth copying: never use a comparative without a number beside it, and state
what a thing does not do as plainly as what it does.

---

*Portions of this guide describe conventions inferred from merged pull requests and dated
reports rather than from a previously stated rule — specifically the six-part report
structure, the unchecked-box Test plan, and the "Alternatives I rejected" section.
Corrections are welcome; the intent is to write down what is already true, not to fix it
in place.*
