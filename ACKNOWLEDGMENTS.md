# Acknowledgments, lineage, and disclaimers

## Disclaimer

> **mlxlearn is not officially associated with scikit-learn or PROBABL, nor with Apple.**

`mlxlearn` is an independent project. The names *scikit-learn* and *sklearn* are used
only descriptively, to identify the third-party library that mlxlearn interoperates
with. The identifier `patch_sklearn` names a function whose sole purpose is to patch
scikit-learn; that is descriptive use of a third-party name, not a claim of endorsement
or affiliation. *MLX* and *Apple silicon* identify Apple's array framework and hardware;
mlxlearn is not an Apple product.

The trademark analysis behind the project name is a careful reading of the published
scikit-learn brand guidelines (February 2025), not legal advice. Naming, disclaimers,
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) require human legal review before
the first public release.

## Lineage

mlxlearn is the successor to a private research repository, `scikit_learn_mps`, which
was itself a fork of Intel's `scikit-learn-intelex`. That history is **not** carried into
this repository.

This repository was bootstrapped from an explicit allowlist rather than by copying a tree
and deleting things afterwards. The following categories never entered the history:

- tree ensembles and their kernels,
- pass-through estimator modules that added no acceleration,
- the `legacy/` compatibility tree,
- the historical dispatcher and device-offload machinery,
- generated C sources and compiled `.so` binaries,
- scikit-learn-intelex examples, notebooks, and documentation,
- all Intel / oneDAL / sklearnex branding.

The audit that produced the allowlist lives in [`phase0/`](phase0/), including a
per-file provenance matrix, an estimator migration matrix, and the recorded behavioral
baseline of the ancestor at revision `720053bdf9b7a377ee45bd5bd573bc09a7df1743`.

### How the 0.1.0 code was written

Every module under `src/mlxlearn/` in the 0.1.0 surface was **newly written** for this
repository. The ancestor repository was consulted as a *behavioral reference* — for the
sklearn-compatible parameter surface, the numerical hazards it had already discovered,
and the shape of its blocked-distance strategy — but no source file was copied into
this history. See [`phase0/attestation.md`](phase0/attestation.md) for the per-module
authorship attestation.

Where an idea is traceable to the ancestor, it is credited in the module docstring.

## Upstream projects mlxlearn depends on or interoperates with

- **scikit-learn** (BSD-3-Clause) — the API mlxlearn conforms to and the reference
  implementation it is tested against. Estimator parameter names, attribute names, and
  documented semantics follow scikit-learn's public API by design; that API compatibility
  is the point of the project.
- **MLX** (MIT) — Apple's array framework, the compute backend.
- **NumPy** (BSD-3-Clause) — the array boundary for 0.1.0.

Full notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## A note on scrubbing

Removing a vendor's branding is not the same as removing a vendor's attribution. This
repository removes the former and preserves the latter: any file with a genuine upstream
lineage carries the upstream notice plus a modification line, and is listed in
`THIRD_PARTY_NOTICES.md`. As of 0.1.0a1 no such file exists, because nothing was copied.
