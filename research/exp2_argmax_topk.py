"""Experiment 2: nearest slot (argmax) and top-k under a fixed query budget.

Durr-Hoyer and k-maxima cannot tell when they are done, so they run until
a budget B is spent and return their current best.  The fair comparison is
therefore P(correct) as a function of B for each method:

  quantum   Durr-Hoyer argmax / k-maxima with budget B
  classical scan B random slots and return the best seen (P = B/N for the
            argmax; for top-k the chance all k are inside the sample)

We sweep B = c * sqrt(N) (argmax) and B = c * sqrt(kN) (top-k) and report
the smallest c reaching 90% success, which is the constant that matters,
and the crossover N where that budget drops below N.

Problem sources as in experiment 1 (synthetic uniform values, or real
similarity rows restricted to N slots).  Writes exp2_budget.csv.

    python research/exp2_argmax_topk.py [--trials 100]
"""

import argparse
import csv

import numpy as np

from common import out_path, real_similarity_rows, subsample_slots
from qmem import CountingOracle, durr_hoyer_argmax, k_maxima

NS = (64, 256, 1024, 4096)
CS = (1, 2, 3, 4, 6, 8, 12, 16, 22.5)
K = 16


def classical_partial_scan(values: np.ndarray, budget: int, rng: np.random.Generator,
                           k: int = 1) -> bool:
    """Scan ``budget`` random slots; correct iff the true top-k lie inside the sample."""
    budget = min(budget, values.shape[0])
    seen = rng.choice(values.shape[0], budget, replace=False)
    top = np.argsort(-values)[:k]
    return bool(np.isin(top, seen).all())


def main(trials: int) -> None:
    rng = np.random.default_rng(1)
    rows, _ = real_similarity_rows()
    out = out_path("exp2_budget.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "task", "N", "c", "budget", "quantum_success", "classical_success"])
        for source in ("synthetic", "real"):
            for task in ("argmax", f"top{K}"):
                for n in NS:
                    for c in CS:
                        scale = np.sqrt(n) if task == "argmax" else np.sqrt(K * n)
                        budget = int(c * scale)
                        q_ok = cl_ok = 0
                        for _ in range(trials):
                            if source == "synthetic":
                                values = rng.random(n)
                            else:
                                values = subsample_slots(rows[rng.integers(rows.shape[0])], n, rng)
                            oracle = CountingOracle(values)
                            if task == "argmax":
                                q_ok += int(durr_hoyer_argmax(oracle, rng, max_queries=budget) == oracle.argmax())
                                cl_ok += int(classical_partial_scan(values, budget, rng))
                            else:
                                found = k_maxima(oracle, K, rng, max_queries=budget)
                                q_ok += int(set(found.tolist()) == set(oracle.top_k(K).tolist()))
                                cl_ok += int(classical_partial_scan(values, budget, rng, k=K))
                        w.writerow([source, task, n, c, budget, f"{q_ok / trials:.3f}", f"{cl_ok / trials:.3f}"])
                        print(f"{source:9s} {task:6s} N={n:5d} c={c:5.1f} B={budget:5d}"
                              f"  quantum {q_ok / trials:.2f}  classical {cl_ok / trials:.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    main(parser.parse_args().trials)
