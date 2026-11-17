"""Experiment 3: estimate p(real | q) by amplitude estimation vs sampling.

For each query, the softmax weights over N slots define the state psi and
the amplitude a = p(real | q).  We run maximum-likelihood QAE with growing
schedules and compare |estimate - a| against classical sampling at the same
number of uses of the state-preparation unitary A.  Expected: error falls
like 1/uses for QAE and 1/sqrt(uses) classically.

Also records the state-preparation honesty figure of report 03: the
multiplier sqrt(1 / p_eff) with p_eff = Z / (N w_max), which is what a
coherent preparation of psi would cost in similarity-oracle calls per use.

Sources: "random" (random unit keys in R^128, random query, random labels)
and "real" (trained memory, MNIST queries).  Writes exp3_qae.csv and
exp3_prep.csv.

    python research/exp3_posterior_qae.py [--queries 20] [--repeats 20]
"""

import argparse
import csv

import numpy as np

from common import out_path, real_similarity_rows
from qmem.amplitude import (classical_estimate, good_probability_after, mlae_estimate,
                            posterior_state)

KAPPA = 10.0
SHOTS = 32
SCHEDULES = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 4], [0, 1, 2, 4, 8], [0, 1, 2, 4, 8, 16],
             [0, 1, 2, 4, 8, 16, 32], [0, 1, 2, 4, 8, 16, 32, 64], [0, 1, 2, 4, 8, 16, 32, 64, 128]]


def random_instance(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    keys = rng.standard_normal((n, 128))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    q = rng.standard_normal(128)
    q /= np.linalg.norm(q)
    return keys @ q, rng.random(n) < 0.5


def main(n_queries: int, repeats: int) -> None:
    rng = np.random.default_rng(2)
    rows, is_real = real_similarity_rows(n_queries)

    # Sanity: the explicit Grover operator reproduces sin^2((2m+1) theta).
    sims, good = random_instance(256, rng)
    psi = posterior_state(np.exp(KAPPA * sims))
    a = float((psi[good] ** 2).sum())
    for m in (1, 3, 7):
        assert abs(good_probability_after(psi, good, m) - np.sin((2 * m + 1) * np.arcsin(np.sqrt(a))) ** 2) < 1e-9
    print("Grover operator check passed")

    qae_out = out_path("exp3_qae.csv")
    with qae_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "uses_of_A", "qae_median_abs_err", "classical_median_abs_err", "a_mean"])
        for source in ("random", "real"):
            per_budget = {}
            a_values = []
            for qi in range(n_queries):
                if source == "random":
                    sims, good = random_instance(4096, rng)
                else:
                    sims, good = rows[qi], is_real
                weights = np.exp(KAPPA * sims)
                a = float(weights[good].sum() / weights.sum())
                a_values.append(a)
                for schedule in SCHEDULES:
                    for _ in range(repeats):
                        est, uses = mlae_estimate(a, schedule, SHOTS, rng)
                        cls = classical_estimate(a, uses, rng)
                        per_budget.setdefault(uses, ([], []))
                        per_budget[uses][0].append(abs(est - a))
                        per_budget[uses][1].append(abs(cls - a))
            for uses in sorted(per_budget):
                q_err, c_err = per_budget[uses]
                w.writerow([source, uses, f"{np.median(q_err):.5f}", f"{np.median(c_err):.5f}",
                            f"{np.mean(a_values):.3f}"])
                print(f"{source:6s} uses={uses:6d}  QAE {np.median(q_err):.5f}  classical {np.median(c_err):.5f}")

    prep_out = out_path("exp3_prep.csv")
    with prep_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "N", "p_eff_median", "prep_multiplier_median"])
        for source in ("random", "real"):
            for n in (64, 256, 1024, 4096):
                p_effs = []
                for qi in range(n_queries):
                    if source == "random":
                        sims, _ = random_instance(n, rng)
                    else:
                        sims = rng.choice(rows[qi], n, replace=False) if n < 4096 else rows[qi]
                    wts = np.exp(KAPPA * sims)
                    p_effs.append(wts.sum() / (n * wts.max()))
                med = float(np.median(p_effs))
                w.writerow([source, n, f"{med:.4f}", f"{np.sqrt(1 / med):.2f}"])
                print(f"{source:6s} N={n:5d}  p_eff {med:.4f}  sqrt(1/p_eff) {np.sqrt(1 / med):.1f}")
    print(f"-> {qae_out}\n-> {prep_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    main(args.queries, args.repeats)
