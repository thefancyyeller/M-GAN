"""Amplitude estimation of the memory posterior p(real | q).

The posterior is a = sum_{i real} w_i / Z with w_i = exp(kappa * s_i).  If a
unitary A prepares psi = sum_i sqrt(w_i / Z) |i>, then a is the squared
norm of psi on the "real" indices, i.e. an amplitude, and amplitude
estimation (Brassard, Hoyer, Mosca, Tapp 2002) finds it to additive error
eps with O(1/eps) uses of A, against O(1/eps^2) classical samples.

Accounting: one **use of A** is the unit.  A classical sample draws one
index from w/Z, which also costs one use of A (or one draw from the same
distribution), so the two sides are compared on the same axis.  A shot with
m Grover iterations costs 2m + 1 uses of A (A once, then A^dag and A inside
each Q).  Preparing A itself is the state-prep cost discussed in
docs/research/03-amplitude-estimation.md and is not counted here.

Implemented: the Grover operator for a known psi (to verify the rotation
formula) and maximum-likelihood AE (Suzuki et al. 2020), which needs no QFT
and whose shots are exactly Binomial(n, sin^2((2m+1) theta)).
"""

import numpy as np


def posterior_state(weights: np.ndarray) -> np.ndarray:
    """psi_i = sqrt(w_i / Z): the state A prepares."""
    return np.sqrt(weights / weights.sum())


def grover_operator(psi: np.ndarray, good: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply Q = -(I - 2 psi psi^T)(I - 2 P_good) to vector x, O(N)."""
    v = x.copy()
    v[good] *= -1.0                       # S_chi
    return -(v - 2.0 * psi * (psi @ v))   # -A S_0 A^dag


def good_probability_after(psi: np.ndarray, good: np.ndarray, m: int) -> float:
    """Explicit statevector check of sin^2((2m+1) theta)."""
    x = psi.copy()
    for _ in range(m):
        x = grover_operator(psi, good, x)
    return float((x[good] ** 2).sum())


def mlae_estimate(a_true: float, schedule: list[int], shots: int,
                  rng: np.random.Generator, grid: int = 200_000) -> tuple[float, int]:
    """Maximum-likelihood amplitude estimation.

    For each m in ``schedule`` draw ``shots`` outcomes from
    Binomial(shots, sin^2((2m+1) theta)) (exact for an ideal simulator), then
    maximise the joint likelihood over theta on a grid.

    Returns (estimate of a, uses of A = shots * sum(2m + 1)).
    """
    theta = np.arcsin(np.sqrt(a_true))
    hits = [rng.binomial(shots, np.sin((2 * m + 1) * theta) ** 2) for m in schedule]
    thetas = np.linspace(1e-6, np.pi / 2 - 1e-6, grid)
    loglik = np.zeros_like(thetas)
    for m, h in zip(schedule, hits):
        s2 = np.sin((2 * m + 1) * thetas) ** 2
        loglik += h * np.log(s2 + 1e-300) + (shots - h) * np.log(1 - s2 + 1e-300)
    best = thetas[np.argmax(loglik)]
    uses = shots * sum(2 * m + 1 for m in schedule)
    return float(np.sin(best) ** 2), uses


def classical_estimate(a_true: float, uses: int, rng: np.random.Generator) -> float:
    """Draw ``uses`` indices from w/Z and return the real fraction."""
    return float(rng.binomial(uses, a_true) / uses)
