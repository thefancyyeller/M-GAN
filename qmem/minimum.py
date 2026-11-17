"""Durr-Hoyer maximum finding and the k-maxima extension.

Durr & Hoyer 1996 (quant-ph/9607014): keep a current best index y; use BBHT
to find any index whose value beats values[y]; repeat until BBHT fails.
Expected O(sqrt(N)) queries with success probability >= 1/2 when stopped
after about 22.5 sqrt(N) + 1.4 log^2 N queries; we instead let BBHT's own
"no solution" budget end the search, which is simpler and in practice
gives higher success at similar cost (both are reported).

k-maxima (Durr, Heiligman, Hoyer, Mhalla 2004, quant-ph/0401091): keep the
best-k set; search for any index beating its minimum; replace; repeat.
O(sqrt(kN)) queries.
"""

import numpy as np

from .grover import bbht_search
from .oracle import CountingOracle


def durr_hoyer_argmax(oracle: CountingOracle, rng: np.random.Generator,
                      max_queries: int | None = None) -> int:
    """Return an index believed to hold the maximum value.

    ``max_queries`` caps the total; the default is the Durr-Hoyer bound
    22.5 sqrt(N) + 1.4 lg^2 N, after which the current best is returned.
    """
    n = oracle.n
    if max_queries is None:
        max_queries = int(22.5 * np.sqrt(n) + 1.4 * np.log2(n) ** 2)
    start = oracle.queries
    best = int(rng.integers(n))
    best_value = oracle.value(best)
    while oracle.queries - start < max_queries:
        remaining = max_queries - (oracle.queries - start)
        found = bbht_search(oracle, best_value, rng, max_queries=remaining)
        if found is None:
            break
        best, best_value = found, oracle.value(found)
    return best


def k_maxima(oracle: CountingOracle, k: int, rng: np.random.Generator,
             max_queries: int | None = None) -> np.ndarray:
    """Return k indices believed to hold the k largest values (any order).

    Keeps a set S of k candidates; repeatedly BBHT-searches for an index
    whose value beats min(S) and is not already in S (the oracle threshold
    handles the first condition, the second is a classical check), swaps it
    in, and stops when BBHT finds nothing.  Default budget 22.5 sqrt(kN).
    """
    n = oracle.n
    if max_queries is None:
        max_queries = int(22.5 * np.sqrt(k * n) + 1.4 * np.log2(n) ** 2)
    start = oracle.queries
    members = list(rng.choice(n, size=k, replace=False))
    values = {i: oracle.value(i) for i in members}
    while oracle.queries - start < max_queries:
        worst = min(members, key=values.get)
        remaining = max_queries - (oracle.queries - start)
        found = bbht_search(oracle, values[worst], rng, max_queries=remaining)
        if found is None:
            break
        if found in values:
            continue                      # measured a current member; try again
        members.remove(worst)
        del values[worst]
        members.append(found)
        values[found] = oracle.value(found)
    return np.array(members)
