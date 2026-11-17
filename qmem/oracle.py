"""The query-counting oracle shared by classical and quantum algorithms.

Accounting convention (the same for both sides):

* The oracle owns a fixed array ``values[i]`` (e.g. the cosine similarity
  of one query to slot ``i``, or the age of slot ``i``).
* One **query** is one evaluation of the predicate ``values[i] > threshold``
  for one classical index ``i``, *or* one application of the phase-flip
  unitary O_t |i> = (-1)^[values[i] > t] |i> to the whole superposition.
  The quantum side therefore pays one query per Grover iteration however
  many indices are in superposition; that is the standard black-box model.
* Preparing the oracle (loading the keys, computing the similarities that
  ``values`` holds) is *not* counted here.  It is the QRAM / state-prep
  cost and is reported separately in the research plan.
"""

import numpy as np


class CountingOracle:
    """Wraps a value array and counts every threshold query made against it."""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.float64)
        self.n = self.values.shape[0]
        self.queries = 0

    # --- classical access -------------------------------------------------

    def above(self, i: int, threshold: float) -> bool:
        """Classical query: is values[i] > threshold?"""
        self.queries += 1
        return bool(self.values[i] > threshold)

    def value(self, i: int) -> float:
        """Classical query that reveals the value itself (a scan step)."""
        self.queries += 1
        return float(self.values[i])

    # --- quantum access ---------------------------------------------------

    def phase_flip(self, state: np.ndarray, threshold: float) -> np.ndarray:
        """Quantum query: apply O_t to an N-dim amplitude vector."""
        self.queries += 1
        signs = np.where(self.values > threshold, -1.0, 1.0)
        return state * signs

    # --- ground truth (not counted; used only to score results) -----------

    def marked(self, threshold: float) -> np.ndarray:
        return np.flatnonzero(self.values > threshold)

    def argmax(self) -> int:
        return int(np.argmax(self.values))

    def top_k(self, k: int) -> np.ndarray:
        return np.argsort(-self.values, kind="stable")[:k]
