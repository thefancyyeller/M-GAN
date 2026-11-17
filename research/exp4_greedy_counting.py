"""Experiment 4: the "greedy amplitude-estimation" idea for top-k retrieval.

Hypothesis (owner's idea, read charitably): instead of k-maxima, binary
search a similarity threshold tau so that about k slots exceed it, using
quantum counting at each step, then collect the slots above tau with
repeated BBHT searches.  Each counting step estimates the marked fraction
t/N as an amplitude (uniform superposition, good = marked), so it costs
O(sqrt(N/t)/eps_rel) queries; the collection phase costs about
k * sqrt(N/k) = sqrt(kN).  Total should scale like sqrt(N) * polylog, to be
compared with k-maxima (experiment 2) and the classical N scan.

Counting is simulated with maximum-likelihood AE on a = t/N (exact
Binomial shots).  The binary search runs over the sorted-similarity index
space: tau is the value at a candidate rank, so each step halves the rank
interval.  Writes exp4_greedy.csv.

    python research/exp4_greedy_counting.py [--trials 50]
"""

import argparse
import csv

import numpy as np

from common import out_path, real_similarity_rows, subsample_slots
from qmem import CountingOracle, bbht_search, k_maxima
from qmem.amplitude import mlae_estimate

NS = (64, 256, 1024, 4096)
K = 16
COUNT_SCHEDULE = [0, 1, 2, 4, 8]      # 32 shots each: 5 * 32 * ... uses, rel. error ~ few %
COUNT_SHOTS = 16


def quantum_count(oracle: CountingOracle, threshold: float, rng: np.random.Generator) -> tuple[float, int]:
    """Estimate the number of slots above threshold; returns (estimate, queries)."""
    t = oracle.marked(threshold).size                       # ground truth drives the exact simulator
    est, uses = mlae_estimate(t / oracle.n, COUNT_SCHEDULE, COUNT_SHOTS, rng)
    oracle.queries += uses
    return est * oracle.n, uses


def greedy_topk(oracle: CountingOracle, k: int, rng: np.random.Generator) -> np.ndarray:
    """Binary-search a threshold admitting ~k slots, then collect them."""
    values = oracle.values
    lo, hi = 0, oracle.n                                    # rank interval: tau = rank-th largest
    order = np.sort(values)[::-1]
    # Binary search on rank until the counted number of slots above tau is in [k, 2k).
    for _ in range(int(np.ceil(np.log2(oracle.n))) + 2):
        mid = (lo + hi) // 2
        tau = float(order[mid]) if mid < oracle.n else -np.inf
        count, _ = quantum_count(oracle, tau, rng)
        if count < k:
            lo = mid + 1
        elif count >= 2 * k:
            hi = mid
        else:
            break
    tau = float(order[min(mid, oracle.n - 1)])
    # Collect everything above tau: BBHT until it fails, excluding found items classically.
    found: list[int] = []
    while True:
        excluded = np.isin(np.arange(oracle.n), found)
        sub = CountingOracle(np.where(excluded, -np.inf, values))
        i = bbht_search(sub, tau, rng, max_queries=int(3 * np.sqrt(oracle.n)) + 5)
        oracle.queries += sub.queries
        if i is None:
            break
        found.append(i)
    found_arr = np.array(found, dtype=int)
    # Keep the best k of what was collected (one classical query each to read values).
    oracle.queries += found_arr.size
    return found_arr[np.argsort(-values[found_arr])][:k] if found_arr.size else found_arr


def main(trials: int) -> None:
    rng = np.random.default_rng(4)
    rows, _ = real_similarity_rows()
    out = out_path("exp4_greedy.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "N", "method", "mean_queries", "exact_rate", "mean_overlap"])
        for source in ("synthetic", "real"):
            for n in NS:
                stats = {"greedy_counting": ([], [], []), "k_maxima": ([], [], [])}
                for _ in range(trials):
                    values = rng.random(n) if source == "synthetic" else \
                        subsample_slots(rows[rng.integers(rows.shape[0])], n, rng)
                    truth = set(np.argsort(-values)[:K].tolist())
                    for method, (qs, exact, overlap) in stats.items():
                        oracle = CountingOracle(values)
                        if method == "greedy_counting":
                            got = set(greedy_topk(oracle, K, rng).tolist())
                        else:
                            got = set(k_maxima(oracle, K, rng, max_queries=int(6 * np.sqrt(K * n))).tolist())
                        qs.append(oracle.queries)
                        exact.append(got == truth)
                        overlap.append(len(got & truth) / K)
                for method, (qs, exact, overlap) in stats.items():
                    w.writerow([source, n, method, f"{np.mean(qs):.1f}", f"{np.mean(exact):.3f}", f"{np.mean(overlap):.3f}"])
                    print(f"{source:9s} N={n:5d} {method:16s} queries {np.mean(qs):8.1f}  exact {np.mean(exact):.2f}"
                          f"  overlap {np.mean(overlap):.2f}   (N={n}, sqrt(kN)={np.sqrt(K * n):.0f})")
    print(f"-> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    main(parser.parse_args().trials)
