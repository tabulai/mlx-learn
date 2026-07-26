# mlxlearn

`mlxlearn` is the planned successor to `scikit_learn_mps`: a
scikit-learn-compatible classical machine-learning library accelerated with MLX
on Apple silicon.

The repository is currently in **Phase 0 (freeze and audit)**. No runtime code
has been copied yet. This is intentional: candidate source files must be
classified, the historical behavior must be recorded, and migration decisions
must be explicit before the new package is bootstrapped.

Phase 0 is reproducible with:

```bash
python3 tools/phase0_audit.py \
  --source /Users/tunguz/Programming/scikit_learn_mps \
  --upstream /Users/tunguz/Programming/sklearn_repos/scikit-learn-intelex

python3 tools/run_phase0_baseline.py \
  --source /Users/tunguz/Programming/scikit_learn_mps
```

See [`phase0/README.md`](phase0/README.md) for the artifacts and exit criteria.
