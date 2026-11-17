"""Exact amplitude-level Grover search and the BBHT unknown-M schedule.

The state is a real N-vector of amplitudes over slot indices.  One Grover
iteration is G = D O_t, with O_t the oracle phase flip (one query) and
D = 2|s><s| - I the diffusion about the uniform state.  Simulating the
N-vector directly is exact and costs O(N) per iteration, which is nothing
for N <= 4096.

References: Grover 1996 (quant-ph/9605043); Boyer, Brassard, Hoyer, Tapp
1998 (quant-ph/9605034) for the unknown-M exponential schedule.
"""

import numpy as np

from .oracle import CountingOracle


def _diffuse(state: np.ndarray) -> np.ndarray:
    mean = state.mean()
    return 2.0 * mean - state


def grover_iterations(oracle: CountingOracle, threshold: float, j: int,
                      rng: np.random.Generator) -> int:
    """Run j Grover iterations from the uniform state and measure.

    Returns the measured index.  Costs exactly j oracle queries.
    """
    state = np.full(oracle.n, 1.0 / np.sqrt(oracle.n))
    for _ in range(j):
        state = _diffuse(oracle.phase_flip(state, threshold))
    probs = state ** 2
    probs /= probs.sum()
    return int(rng.choice(oracle.n, p=probs))


def bbht_search(oracle: CountingOracle, threshold: float, rng: np.random.Generator,
                lam: float = 6 / 5, max_queries: int | None = None) -> int | None:
    """Find an index with values[i] > threshold when the number of such
    indices is unknown (BBHT, Theorem 3).

    Loop: pick j uniformly in [0, m), run j Grover iterations, measure, and
    check the result with one classical query.  On failure grow
    m <- min(lam * m, sqrt(N)).  Expected cost O(sqrt(N/M)) when M >= 1
    marked indices exist.

    Returns the found index, or None once ``max_queries`` is exhausted
    (which is how a caller detects "probably no marked index").
    """
    if max_queries is None:
        max_queries = int(9 * np.sqrt(oracle.n)) + 10     # generous "no solution" budget
    start = oracle.queries
    m = 1.0
    while oracle.queries - start < max_queries:
        j = int(rng.integers(0, int(np.ceil(m))))
        i = grover_iterations(oracle, threshold, j, rng)
        if oracle.above(i, threshold):
            return i
        m = min(lam * m, np.sqrt(oracle.n))
    return None
