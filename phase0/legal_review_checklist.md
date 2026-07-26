# Human legal review checklist (blocking gate before first public release)

Automated tooling can surface evidence. It cannot make authorship or trademark
determinations. Publishing `mlxlearn` to PyPI or announcing it publicly is gated on a
lawyer signing off on the items below.

Nothing in this repository, in `ACKNOWLEDGMENTS.md`, or in `THIRD_PARTY_NOTICES.md` is
legal advice.

## Status

| # | Item | Prepared artifact | Reviewer | Date | Verdict |
|---|---|---|---|---|---|
| 1 | Project name `mlxlearn` under the scikit-learn brand guidelines (Feb 2025) | `ACKNOWLEDGMENTS.md` §Disclaimer, `THIRD_PARTY_NOTICES.md` §4 | | | ☐ |
| 2 | `patch_sklearn` as a public identifier (nominative/descriptive use) | plan §3; `src/mlxlearn/patching/` | | | ☐ |
| 3 | Disclaimer wording and its placement (README, docs, package metadata) | `README.md`, `ACKNOWLEDGMENTS.md` | | | ☐ |
| 4 | `THIRD_PARTY_NOTICES.md` completeness and accuracy | that file | | | ☐ |
| 5 | Provenance report and per-file evidence | `generated/provenance.csv`, `generated/audit_summary.md` | | | ☐ |
| 6 | Authorship attestation for the shipped surface | `attestation.md` | | | ☐ |
| 7 | Apache-2.0 outbound license vs. BSD-3-Clause inbound obligations (scikit-learn `COPYING`) | `LICENSE`, `THIRD_PARTY_NOTICES.md` §2 | | | ☐ |
| 8 | Dependency license scan (no copyleft in the runtime dependency closure) | CI job `compliance` → `tools/check_dependency_licenses.py` | | | ☐ |
| 9 | Apple / MLX / Metal naming and the non-affiliation statement | `THIRD_PARTY_NOTICES.md` §4 | | | ☐ |
| 10 | PyPI project name and description copy | `pyproject.toml` | | | ☐ |

## Questions to put to the reviewer explicitly

1. Does `patch_sklearn` as an exported public symbol read as nominative use, or does it
   need renaming (e.g. `patch_estimators`) with `patch_sklearn` kept only as a documented
   alias?
2. Is the disclaimer sufficient in package metadata alone, or must it appear in the
   rendered PyPI long description and the docs landing page?
3. mlxlearn deliberately mirrors scikit-learn's parameter and attribute names. Confirm
   the position taken in `THIRD_PARTY_NOTICES.md` §2 — that this is interface
   compatibility, not derivation — and whether stronger attribution is warranted.
4. The ancestor repository was a fork of Intel's `scikit-learn-intelex` (Apache-2.0).
   Nothing was copied into this repository. Confirm that the fresh-history bootstrap plus
   `attestation.md` is a sufficient record, or specify what more is needed.
5. Should `phase0/` ship inside the sdist, or stay repository-only?

## Enforcement

CI runs the mechanical parts of this checklist on every push
(`.github/workflows/ci.yml` → job `compliance`). The mechanical checks passing is a
necessary condition for release, never a sufficient one — items 1–7 and 9–10 above require
a human.
