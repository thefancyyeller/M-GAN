"""Classical baselines, charged through the same CountingOracle."""

import numpy as np

from .oracle import CountingOracle


def linear_scan_argmax(oracle: CountingOracle) -> int:
    """N queries, always correct."""
    best, best_value = 0, oracle.value(0)
    for i in range(1, oracle.n):
        v = oracle.value(i)
        if v > best_value:
            best, best_value = i, v
    return best


def linear_scan_threshold(oracle: CountingOracle, threshold: float,
                          rng: np.random.Generator) -> int | None:
    """Scan in a random order until an index above threshold is found."""
    for i in rng.permutation(oracle.n):
        if oracle.above(int(i), threshold):
            return int(i)
    return None


def random_probe_threshold(oracle: CountingOracle, threshold: float,
                           rng: np.random.Generator, max_queries: int) -> int | None:
    """Sample indices with replacement; expected N/M queries."""
    for _ in range(max_queries):
        i = int(rng.integers(oracle.n))
        if oracle.above(i, threshold):
            return i
    return None
