# Authorship attestation for the migrated surface

The Phase 0 provenance matrix (`generated/provenance.csv`) records *evidence*: path matches,
header matches, and text-similarity scores against `scikit-learn-intelex`. As
[`README.md`](README.md) states, the absence of those signals is not proof of independent
authorship. This file records the second half of the requirement from the refactoring plan
§8.2 — a per-module authorship statement for everything that actually ships.

## Method used for the 0.1.0 surface

The Phase 0 audit left 12 files with a `port-candidate` disposition, of which only
`mpsbackend/neighbors/__init__.py` and `mpsbackend/neighbors/neighbors.py` had completed
technical review. Rather than block on the remaining review queue, the 0.1.0 surface was
built with a stricter rule:

> **No file was copied. Every module under `src/mlxlearn/` was newly written.**

The ancestor was used only as a *behavioral reference*, in the sense the audit policy
assigns to the `rewrite` and `behavioral-reference` dispositions: to learn which
sklearn parameters must be honored, which numerical hazards had already been discovered,
and which fallbacks are required. This makes the pending `port-candidate` review queue
moot for the shipped surface: there is nothing ported to review.

Ideas — as opposed to expression — that are traceable to the ancestor are credited in the
relevant module docstring. Two are worth naming here:

- **Blocked distance evaluation with a two-level query/train tiling** for exact KNN, so that
  the `n_query × n_train` distance matrix is never materialized. The tiling *strategy* comes
  from the ancestor's `mpsbackend/neighbors/neighbors.py` (technical-reviewed, blame-attributed
  to the repository owner); the mlxlearn implementation is new code with a different top-k
  primitive (`mx.argpartition` instead of an iterated `argmin` loop) and a different merge.
- **Recomputing the distances of the selected candidates** from the original vectors instead
  of trusting the expanded `‖q‖² + ‖t‖² − 2q·t` values, because that expansion loses precision
  catastrophically in fp32 for near-duplicate rows. The hazard was discovered in the ancestor;
  the fix is standard practice and is reimplemented here.

## Per-module attestation

| mlxlearn module | Author | Basis | Ancestor relationship |
|---|---|---|---|
| `_common/*` | newly written | scikit-learn public API + MLX docs | none — the ancestor's `_config.py`/`_device_manager.py` were `rewrite`-dispositioned and not consulted for expression |
| `_kernels/distance.py` | newly written | standard blocked-GEMM distance formulation | tiling *strategy* credited above |
| `_solvers/*` | newly written | published algorithms (L-BFGS; SMO, Platt 1998 / Fan–Chen–Lin 2005) | none |
| `neighbors/*` | newly written | scikit-learn `neighbors` public API | parameter surface and fallback conditions informed by the ancestor's wrappers |
| `linear_model/logistic.py` | newly written | scikit-learn `LogisticRegression` public API | parameter surface only |
| `svm/classes.py` | newly written | scikit-learn `SVC` public API; LIBSVM's published algorithm | parameter surface only |
| `patching/*` | newly written | Python import semantics | the ancestor's `dispatcher.py` was `rewrite`-dispositioned; its *taxonomy* of failure modes informed §2 of the plan, its code did not enter this repository |
| `tests/*` | newly written | scikit-learn as the parity oracle | none |
| `benchmarks/*` | newly written | — | none |
| `tools/*`, `phase0/*` | authored in this repository during Phase 0 | — | none |

## Scope of this statement

This is a technical attestation by the engineer who wrote the code, recording what was and
was not consulted. It is a technical record, not a legal conclusion.
