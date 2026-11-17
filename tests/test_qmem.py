"""Sanity tests for the amplitude-level quantum simulators in ``qmem``."""

import numpy as np
import pytest

from qmem import (CountingOracle, bbht_search, durr_hoyer_argmax, grover_iterations,
                  k_maxima, linear_scan_argmax)
from qmem.amplitude import (classical_estimate, good_probability_after, mlae_estimate,
                            posterior_state)


@pytest.fixture
def rng():
    return np.random.default_rng(123)


def test_grover_known_m_hits_optimal_iteration(rng):
    n, m = 1024, 1
    values = np.zeros(n)
    values[17] = 1.0
    oracle = CountingOracle(values)
    j = int(np.floor(np.pi / 4 * np.sqrt(n / m)))          # 25 for N=1024
    hits = sum(grover_iterations(oracle, 0.5, j, rng) == 17 for _ in range(50))
    assert hits >= 48
    assert oracle.queries == 50 * j


def test_bbht_finds_marked_in_sqrt_queries(rng):
    n, m = 1024, 4
    counts = []
    for _ in range(50):
        values = rng.random(n)
        values[rng.choice(n, m, replace=False)] += 1.0
        oracle = CountingOracle(values)
        i = bbht_search(oracle, 1.0, rng)
        assert values[i] > 1.0
        counts.append(oracle.queries)
    assert np.mean(counts) < 4 * np.sqrt(n / m)             # BBHT bound is 4.5 sqrt(N/M)


def test_bbht_returns_none_when_nothing_marked(rng):
    oracle = CountingOracle(np.zeros(64))
    assert bbht_search(oracle, 0.5, rng) is None
    assert oracle.queries <= 9 * 8 + 10 + 8                  # stays within its budget


def test_durr_hoyer_matches_scan(rng):
    n = 256
    ok = 0
    for _ in range(30):
        values = rng.random(n)
        assert linear_scan_argmax(CountingOracle(values)) == int(np.argmax(values))
        ok += durr_hoyer_argmax(CountingOracle(values), rng, max_queries=int(8 * np.sqrt(n))) == int(np.argmax(values))
    assert ok >= 20                                          # well above the 1/2 guarantee


def test_k_maxima_recovers_top_k(rng):
    n, k = 256, 8
    ok = 0
    for _ in range(20):
        values = rng.random(n)
        oracle = CountingOracle(values)
        found = set(k_maxima(oracle, k, rng, max_queries=int(8 * np.sqrt(k * n))).tolist())
        ok += found == set(oracle.top_k(k).tolist())
    assert ok >= 12


def test_grover_operator_rotation_formula(rng):
    weights = np.exp(10 * rng.standard_normal(128) * 0.3)
    good = rng.random(128) < 0.5
    psi = posterior_state(weights)
    a = float((psi[good] ** 2).sum())
    theta = np.arcsin(np.sqrt(a))
    for m in (1, 2, 5):
        assert abs(good_probability_after(psi, good, m) - np.sin((2 * m + 1) * theta) ** 2) < 1e-9


def test_mlae_beats_classical_at_equal_budget(rng):
    a = 0.3
    schedule = [0, 1, 2, 4, 8, 16, 32]
    q_err, c_err = [], []
    for _ in range(20):
        est, uses = mlae_estimate(a, schedule, 32, rng)
        q_err.append(abs(est - a))
        c_err.append(abs(classical_estimate(a, uses, rng) - a))
    assert np.median(q_err) < np.median(c_err)
