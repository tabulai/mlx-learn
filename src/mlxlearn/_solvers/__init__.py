"""Numerical solvers: typed configuration in, typed state out.

Private. A solver never sees an estimator, never touches scikit-learn, and never
reads global configuration. It takes arrays and a frozen config dataclass and
returns a frozen state dataclass. That boundary is what makes the numerics
testable without constructing an estimator, and what stops solver tuning from
leaking into the public API.
"""

from __future__ import annotations

__all__: list[str] = []
