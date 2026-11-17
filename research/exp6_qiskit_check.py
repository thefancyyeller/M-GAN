r"""Experiment 6: gate-level check of the numpy Grover model at N = 16.

Builds the Grover operator in Qiskit from a phase oracle over 4 index
qubits (marked set from a real similarity row thresholded at tau), runs j
iterations on the Aer statevector simulator, and compares the probability of
measuring a marked index with the numpy amplitude model used by ``qmem``.
Also counts oracle applications in the transpiled circuit to confirm the
query accounting.  Run with the Qiskit venv:

    D:\M-GAN\venv-q\Scripts\python.exe research/exp6_qiskit_check.py

Writes exp6_qiskit.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, grover_operator
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qmem import CountingOracle, grover_iterations

N_QUBITS = 4
N = 2 ** N_QUBITS


def phase_oracle(marked: np.ndarray) -> QuantumCircuit:
    """Diagonal gate with -1 on marked basis states, named 'oracle'."""
    diag = np.ones(N)
    diag[marked] = -1.0
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    qc.append(DiagonalGate(diag.tolist()), list(range(N_QUBITS)))
    gate = qc.to_gate(label="oracle")
    gate.name = "oracle"
    return gate


def main() -> None:
    rng = np.random.default_rng(6)
    sim = AerSimulator(method="statevector")
    out_dir = Path(r"D:\M-GAN\outputs\research")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in (1, 2, 4):
        values = rng.random(N)
        marked = rng.choice(N, m, replace=False)
        values[marked] += 1.0
        oracle_gate = phase_oracle(marked)
        grover = grover_operator(oracle_gate)                       # includes the diffusion
        theta = np.arcsin(np.sqrt(m / N))
        for j in range(0, 5):
            qc = QuantumCircuit(N_QUBITS)
            qc.h(range(N_QUBITS))
            for _ in range(j):
                qc.append(grover, range(N_QUBITS))
            # Exact statevector probability of the marked set.
            probs = Statevector(qc).probabilities()
            p_gate = float(probs[marked].sum())
            # Aer shots.
            qc_meas = qc.copy()
            qc_meas.measure_all()
            counts = sim.run(transpile(qc_meas, sim), shots=20_000).result().get_counts()
            hits = sum(c for bits, c in counts.items() if int(bits, 2) in set(marked.tolist()))
            p_shots = hits / 20_000
            # numpy model: sample many measurements through qmem's simulator.
            o = CountingOracle(values)
            p_numpy = np.mean([grover_iterations(o, 1.0, j, rng) in set(marked.tolist()) for _ in range(2000)])
            oracle_count = qc.decompose(reps=1).count_ops().get("oracle", 0)   # one per Grover iteration
            p_formula = float(np.sin((2 * j + 1) * theta) ** 2)
            rows.append([m, j, f"{p_formula:.5f}", f"{p_gate:.5f}", f"{p_shots:.4f}", f"{p_numpy:.4f}",
                         oracle_count, o.queries // 2000])
            print(f"M={m} j={j}: formula {p_formula:.4f}  qiskit exact {p_gate:.4f}  aer shots {p_shots:.4f}"
                  f"  numpy {p_numpy:.4f}  oracle gates {oracle_count}  numpy queries/run {o.queries // 2000}")
            assert abs(p_gate - p_formula) < 1e-9
    out = out_dir / "exp6_qiskit.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "iterations", "p_formula", "p_qiskit_exact", "p_aer_shots", "p_numpy",
                    "oracle_gates", "numpy_queries"])
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
