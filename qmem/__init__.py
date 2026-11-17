"""Quantum query-complexity experiments for the memoryGAN memory.

Everything here is an *exact* amplitude-level simulation in numpy: no gates,
no noise, no hardware.  The quantity measured is the number of oracle
queries, counted identically for classical and quantum algorithms through
``CountingOracle``.  See docs/research/PLAN.md for what is and is not
claimed.
"""

from .oracle import CountingOracle
from .grover import grover_iterations, bbht_search
from .minimum import durr_hoyer_argmax, k_maxima
from .classical import linear_scan_argmax, linear_scan_threshold, random_probe_threshold

__all__ = [
    "CountingOracle", "grover_iterations", "bbht_search", "durr_hoyer_argmax",
    "k_maxima", "linear_scan_argmax", "linear_scan_threshold", "random_probe_threshold",
]
