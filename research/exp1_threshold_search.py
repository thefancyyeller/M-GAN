"""Experiment 1: find a slot above a similarity threshold (BBHT vs classical).

The ``write`` rule needs "a slot with the same label whose similarity beats
a threshold"; ``lookup`` can be phrased the same way.  With M of N slots
marked, classical random probing needs N/M expected queries, a scan N/M as
well (random order) with worst case N; BBHT needs O(sqrt(N/M)).

Two problem sources:
  synthetic  random values, M marked slots planted (controls N and M exactly)
  real       similarity rows of MNIST queries to the trained memory, slots
             subsampled to N, threshold = the (M+1)-th largest similarity so
             exactly M slots are marked

Writes exp1_threshold.csv with mean / p95 queries and success per cell.

    python research/exp1_threshold_search.py [--trials 200]
"""

import argparse
import csv

import numpy as np

from common import out_path, real_similarity_rows, subsample_slots
from qmem import CountingOracle, bbht_search, linear_scan_threshold, random_probe_threshold

NS = (64, 256, 1024, 4096)
MS = (1, 2, 4, 16, 64)


def instance(source: str, n: int, m: int, rng: np.random.Generator,
             rows: np.ndarray | None) -> tuple[np.ndarray, float]:
    if source == "synthetic":
        values = rng.random(n)
        values[rng.choice(n, m, replace=False)] += 1.0
        return values, 1.0
    row = rows[rng.integers(rows.shape[0])]
    top = np.argsort(-row)[:m]
    values = subsample_slots(row, n, rng, keep=top)
    threshold = float(np.sort(values)[-m - 1]) if m < n else -np.inf
    return values, threshold


def main(trials: int) -> None:
    rng = np.random.default_rng(0)
    rows, _ = real_similarity_rows()
    out = out_path("exp1_threshold.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "N", "M", "method", "mean_queries", "p95_queries", "success", "sqrt_N_over_M"])
        for source in ("synthetic", "real"):
            for n in NS:
                for m in MS:
                    if m >= n:
                        continue
                    results = {"bbht": [], "random_probe": [], "scan": []}
                    ok = {k: 0 for k in results}
                    for _ in range(trials):
                        values, threshold = instance(source, n, m, rng, rows)
                        for method in results:
                            oracle = CountingOracle(values)
                            if method == "bbht":
                                i = bbht_search(oracle, threshold, rng)
                            elif method == "random_probe":
                                i = random_probe_threshold(oracle, threshold, rng, max_queries=20 * n)
                            else:
                                i = linear_scan_threshold(oracle, threshold, rng)
                            results[method].append(oracle.queries)
                            ok[method] += int(i is not None and values[i] > threshold)
                    for method, qs in results.items():
                        w.writerow([source, n, m, method, f"{np.mean(qs):.2f}",
                                    f"{np.percentile(qs, 95):.1f}", f"{ok[method] / trials:.3f}",
                                    f"{np.sqrt(n / m):.2f}"])
                    print(f"{source:9s} N={n:5d} M={m:3d}  bbht {np.mean(results['bbht']):7.1f}"
                          f"  probe {np.mean(results['random_probe']):8.1f}"
                          f"  scan {np.mean(results['scan']):8.1f}  sqrt(N/M)={np.sqrt(n / m):6.1f}")
    print(f"-> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    main(parser.parse_args().trials)
