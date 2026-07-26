"""Randomness that stays reproducible across the NumPy/MLX boundary.

scikit-learn estimators take ``random_state``; MLX has its own key-based
generator. This module is the single place the two are reconciled, so that a
given ``random_state`` produces the same fit whichever device it runs on.

The rules:

* An estimator's own ``random_state`` wins over the library-level
  ``mlxlearn.set_config(random_state=...)``.
* ``derive_seed`` namespaces sub-streams by a string label. Two independent uses
  inside one fit — initializing a solver and permuting a working set, say — must
  not draw from the same stream, or a change in one silently changes the other.
* Nothing here touches MLX's *global* random state. ``mx.random.seed`` is
  process-global, so a library that called it would reach into unrelated code
  running in the same process.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .config import get_config

__all__ = ["resolve_seed", "derive_seed", "numpy_generator", "mlx_key", "mlx_uniform"]

_SEED_MASK = 0xFFFF_FFFF


def resolve_seed(random_state: int | np.random.RandomState | np.random.Generator | None) -> int:
    """Resolve an estimator's ``random_state`` to a concrete 32-bit seed.

    ``None`` falls through to the library-level configuration, and only if that
    is also ``None`` is a nondeterministic seed drawn — and even then, only when
    ``deterministic=False``. Under the default ``deterministic=True``, a
    ``random_state`` of ``None`` resolves to 0, so that repeating a run repeats
    its result. That differs from scikit-learn, which draws from global entropy;
    the deviation is deliberate, documented, and switchable.
    """
    if isinstance(random_state, (int, np.integer)):
        return int(random_state) & _SEED_MASK
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, _SEED_MASK, dtype=np.int64)) & _SEED_MASK
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, _SEED_MASK)) & _SEED_MASK
    if random_state is not None:
        raise TypeError(
            "random_state must be an int, a numpy RandomState or Generator, or None; "
            f"got {type(random_state).__name__}."
        )

    config = get_config()
    if config.random_state is not None:
        return int(config.random_state) & _SEED_MASK
    if config.deterministic:
        return 0
    return int(np.random.SeedSequence().entropy) & _SEED_MASK  # type: ignore[arg-type]


def derive_seed(seed: int, label: str) -> int:
    """Derive an independent sub-stream seed from ``seed`` and a label.

    Uses BLAKE2b rather than arithmetic on the seed, so that nearby seeds produce
    unrelated sub-streams. ``derive_seed(0, "init")`` and ``derive_seed(1, "init")``
    must not be correlated.
    """
    digest = hashlib.blake2b(
        f"{int(seed) & _SEED_MASK}:{label}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") & _SEED_MASK


def numpy_generator(seed: int, label: str = "") -> np.random.Generator:
    """A NumPy generator for a named sub-stream."""
    return np.random.default_rng(derive_seed(seed, label) if label else seed)


def mlx_key(seed: int, label: str = ""):
    """An MLX PRNG key for a named sub-stream.

    Explicit keys, never ``mx.random.seed``: the global generator belongs to the
    application, not to a library it happens to import.
    """
    import mlx.core as mx

    return mx.random.key(derive_seed(seed, label) if label else seed)


def mlx_uniform(shape: tuple[int, ...], seed: int, label: str = "", *, dtype=None):
    """Uniform ``[0, 1)`` samples from a named sub-stream, as an ``mx.array``."""
    import mlx.core as mx

    return mx.random.uniform(
        shape=shape, dtype=dtype or mx.float32, key=mlx_key(seed, label)
    )
