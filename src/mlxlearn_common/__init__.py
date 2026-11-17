"""Private, dependency-free helpers shared by the host and device interfaces.

Importing this package does not load MLX, NumPy, scikit-learn, or either
estimator interface. In particular, CPU-only neighbor dispatch stays lazy.
"""
