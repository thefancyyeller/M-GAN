# 01 — Grover search and quantum associative memory for fetching stored keys

Scope: `Memory.similarity` / `Memory.lookup` in `mgan/memory.py` (N=4096 slots, D=128 unit-norm keys, k=16 neighbours). Question: can a quantum algorithm reduce the number of *oracle queries* needed to fetch the best-matching slot(s), provable in small simulations? Not addressed: wall-clock, hardware, whole-model training.

## (a) Summary

Grover-type search gives a rigorous, provable quadratic reduction in *oracle query count* for exactly the sub-problem our memory performs: find a slot whose cosine similarity to `q` exceeds a threshold (or the top-k). With M marked slots out of N, the quantum algorithm uses Θ(√(N/M)) oracle calls versus Θ(N/M) expected for classical random probing and Θ(N) for a worst-case linear scan; k-maxima costs O(√(kN)) versus Θ(N). This is the only part of the "quantum associative memory" literature that is both provable and relevant to us. The Ventura–Martinez / Trugenberger memories trade this for exponential *capacity* but pay in probabilistic, low-fidelity retrieval and have no clean mapping to real-valued keys. The gain is conditional on an oracle that evaluates `sim(q, key_i) > τ` in superposition over `i`, which in turn requires either a QRAM holding all N·D amplitudes or re-loading keys at Θ(ND) gate cost per query — the latter erases the speedup. The honest result for the project is a query-complexity gain that can be demonstrated in numpy by counting oracle calls, with QRAM cost listed as an explicit unproven assumption. Nothing in the literature combines a Grover-fetched memory with a GAN; the closest work (quantum-annealer Boltzmann latent spaces for GANs) found no advantage.

## (b) Key papers

| Paper | Takeaway for us |
|---|---|
| Grover, *A fast quantum mechanical algorithm for database search* (STOC 1996), https://arxiv.org/abs/quant-ph/9605043 | O(√N) oracle calls to find 1 marked item of N; the oracle is a phase flip on marked indices. |
| Boyer, Brassard, Høyer, Tapp (BBHT), *Tight bounds on quantum searching*, https://arxiv.org/abs/quant-ph/9605034 | Expected O(√(N/M)) calls when M is *unknown* (exponentially growing random iteration counts); √N is optimal. |
| Dürr & Høyer, *A quantum algorithm for finding the minimum*, https://arxiv.org/abs/quant-ph/9607014 | O(√N) calls to find argmax_i f(i): repeated threshold search with a rising threshold. Exactly "nearest slot". |
| Dürr, Heiligman, Høyer, Mhalla, *Quantum query complexity of some graph problems*, https://arxiv.org/abs/quant-ph/0401091 | Contains the k-minima subroutine: the k best of N in O(√(kN)) queries. Our top-16 lookup. |
| Brassard, Høyer, Mosca, Tapp, *Quantum amplitude amplification and estimation*, https://arxiv.org/abs/quant-ph/0005055 | Amplitude estimation: inner product to additive error ε in O(1/ε) state preparations instead of O(1/ε²) swap-test samples. |
| Wiebe, Kapoor, Svore, *Quantum algorithms for nearest-neighbor methods*, QIC 15 (2015), https://arxiv.org/abs/1401.2142 | Composes coherent inner-product estimation with Dürr–Høyer: nearest neighbour in Õ(√N / ε) queries given oracle access to the vectors. The closest existing template for our lookup. |
| Basheer, Afham, Goyal, *Quantum k-nearest neighbors algorithm*, https://arxiv.org/abs/2003.09187 | Fidelity-based kNN reduced to quantum k-maxima, O(√(kM)) queries; Python code published. |
| Ventura & Martinez, *Quantum associative memory*, Inf. Sci. 124 (2000), https://arxiv.org/abs/quant-ph/9807053 | Stores P binary patterns as a superposition over n qubits (capacity 2^n); recall by modified Grover. Patterns are bit strings, not real vectors. |
| Trugenberger, *Probabilistic quantum memories*, PRL 87, 067901 (2001), https://arxiv.org/abs/quant-ph/0012100; *Phase transitions in quantum pattern recognition*, https://arxiv.org/abs/quant-ph/0204115 | Retrieval by Hamming-distance-weighted amplitudes plus post-selection; probabilistic, needs many repetitions; capacity/quality phase transition. |
| Diamantini & Trugenberger, *Mirror modular cloning and fast quantum associative retrieval*, https://arxiv.org/abs/2206.01644 | Claims retrieval exponentially faster than address-based Grover, but probabilistic with retrieval errors; authors' claim, not independently verified. |
| Rebentrost, Bromley, Weedbrook, Lloyd, *Quantum Hopfield neural network*, PRA 98, 042308 (2018), https://arxiv.org/abs/1710.03599 | Amplitude-encoded Hopfield via HHL-style linear algebra; log(dimension) cost *given* QRAM. |
| Labay-Mora, Fiorelli, Zambrini, Giorgi, *Theoretical framework for quantum associative memories*, QST 10, 035050 (2025), https://arxiv.org/abs/2408.14272 | Open-system/dissipative memories; capacity results, no query-complexity retrieval guarantee. |
| Marsh et al., *High-capacity associative memory in a quantum-optical spin glass*, https://arxiv.org/abs/2509.12202 | Experimental: ~7× Hopfield capacity in 16 spins. Physics platform, not an algorithm we can simulate for lookups. |
| Giovannetti, Lloyd, Maccone, *Quantum random access memory*, PRL 100, 160501 (2008), https://arxiv.org/abs/0708.1879 | Bucket-brigade QRAM: O(log N) *active* gates per query but O(N) physical components; the memory itself must still be built/loaded. |
| Arunachalam et al., *On the robustness of bucket brigade quantum RAM*, https://arxiv.org/abs/1502.03450; Hann et al., *Resilience of QRAM to generic noise*, PRX Quantum 2, 020311, https://arxiv.org/abs/2012.05340 | Conflicting error analyses: with QRAM as a Grover oracle, per-gate error must be o(2^{-n/2}) (Arunachalam); Hann et al. show milder infidelity scaling under generic noise. |
| Jaques & Rattew, *QRAM: a survey and critique*, https://arxiv.org/abs/2305.10310 | Passive, cheap, scalable QRAM is unlikely with current proposals; most asymptotic QML advantage disappears with active (circuit) QRAM. |
| Aaronson, *Read the fine print*, Nat. Phys. 11, 291 (2015), https://doi.org/10.1038/nphys3272 | Speedups vanish if loading classical data or reading out the full answer costs Ω(N). |
| Tang, *A quantum-inspired classical algorithm for recommendation systems*, https://arxiv.org/abs/1807.04271 | If the quantum algorithm assumes sampling-access state preparation, grant the classical algorithm the same and exponential gaps disappear; quadratic (Grover) gaps survive. |
| Zhang, Li, Yuan, *Quantum state preparation with optimal circuit depth*, PRL 129, 230504 (2022), https://arxiv.org/abs/2201.11495 | Arbitrary n-qubit state: Θ(2^n/n) depth without ancillas, Θ(n) depth with O(2^n) ancillas. Loading our N·D=2^19 amplitudes is Θ(ND) work either way. |
| Wilson, Vandal, Hogg, Rieffel, *Quantum-assisted associative adversarial network*, QMI 3, 19 (2021), https://arxiv.org/abs/1904.10573; Urushibata, Ohzeki, Tanaka, https://arxiv.org/abs/2203.15220 | Only "associative memory + GAN" work found: Boltzmann-machine latent prior sampled by a D-Wave annealer. Equivalent to classical on MNIST. |
| Lloyd & Weedbrook, *Quantum GANs*, https://arxiv.org/abs/1804.09139; Niu et al., *Entangling QGANs*, https://arxiv.org/abs/2105.00080 | Quantum GANs exist (EQ-GAN even prepares an approximate QRAM); none uses a Grover-fetched external memory. |

## (c) Complexity statement for `lookup`

Oracle model: one query = one evaluation of `sim(q, key_i)` for one index `i` (classically: one row of `q @ keys.T`; quantumly: one call to a unitary O_τ that flips the phase of |i⟩ iff sim(q, key_i) > τ, or a comparator on an estimated similarity).

* Threshold fetch (find any slot with sim > τ; M such slots, M unknown):
  classical random probing Θ(N/M) expected, linear scan Θ(N) worst case; quantum (BBHT) Θ(√(N/M)) expected. Speedup is √(N/M), not √N: for N=4096, M=16 it is 16× in queries, not 64×.
* Nearest slot (argmax): classical Θ(N); quantum (Dürr–Høyer) O(√N) with success ≥ 1/2, amplified to 1−δ with O(log 1/δ) repeats.
* Top-k (our k=16): classical Θ(N); quantum O(√(kN)) = O(256) vs 4096 for N=4096, k=16.
* Lower bound: Ω(√(N/M)) is tight (BBHT), so no quantum algorithm beats this in the black-box model.

Each quantum oracle call internally costs one *coherent* similarity evaluation to precision ε, i.e. O(1/ε) amplitude-estimation rounds on top of key-loading. The real-logit and slot-posterior (dense softmax over all N) are **not** search problems; Grover gives nothing there, and a top-k approximation of them is what DESIGN.md §5 already found breaks training.

## (d) Assumptions and caveats

1. **QRAM / key loading.** The oracle must map |i⟩ → |i⟩|key_i⟩ in superposition. A bucket-brigade QRAM does this with O(log N) active gates per query but requires O(N) physical cells pre-loaded with the keys; the memory is rewritten every training step (`write`), so the Θ(ND) load cost recurs per step. Without QRAM, preparing a state over all keys costs Θ(ND) gates (Zhang–Li–Yuan), which already exceeds the classical Θ(ND) scan; the speedup then does not survive (Aaronson, Jaques–Rattew). Any claimed gain must be stated *per query, amortised over many queries between writes*, with load cost reported separately.
2. **Similarity precision.** Amplitude-encoding a unit D=128 vector needs 7 qubits; the swap test yields |⟨q|k⟩|² (sign lost, O(1/ε²) shots), the Hadamard test yields the signed ⟨q|k⟩ (needs controlled state-preparation, O(1/ε²) shots, or O(1/ε) with amplitude estimation). Thresholding at τ with margin γ needs ε < γ, so near-ties cost O(1/γ) extra per oracle call. At κ=10 the softmax is sensitive to ~0.1 similarity differences, so ε≈0.02–0.05 is required.
3. **Success probability.** Grover/BBHT/Dürr–Høyer are bounded-error; expected counts assume the standard O(1) repetition. Experiments must report both mean and tail of query counts.
4. **Oracle construction.** A "query" hides a comparator circuit plus a state-prep of q; the comparison to the classical scan is honest only if both sides count one similarity evaluation as one unit.
5. **Differentiability.** Grover returns indices, not gradients. `similarity` must stay differentiable w.r.t. q (memory.py contract); a quantum lookup can only replace the *index-selection* step, with sims then recomputed classically for the k returned slots (O(kD)).
6. **Noise.** Using QRAM as a Grover oracle for N=4096 (n=12) demands per-gate error o(2^{-6}) under Arunachalam et al.'s model; irrelevant for numpy simulation but fatal for near-term hardware claims.
7. **Classical competition.** Classical sublinear-time MIPS/LSH indexes exist, but they are time, not query, bounds; in the black-box query model the classical Θ(N/M) is tight, so the quantum √ gap is real.
8. **Associative-memory line is a different problem.** Ventura–Martinez/Trugenberger store *binary* patterns and retrieve by Hamming distance with post-selection; they do not provide real-valued cosine lookup, and their "exponential capacity" is a different axis from query count. Not applicable without re-designing the key representation.

## (e) Small-scale experiments (oracle-count metrics, numpy/Qiskit, N ≤ 1024)

**E1. Threshold search over slots (BBHT vs classical).** Build N ∈ {64, 256, 1024} random unit keys (D=128), a query q, threshold τ chosen so M ∈ {1, 4, 16, 64} slots are marked. Classical: count similarity evaluations until a marked slot is found, by (i) random probing and (ii) linear scan. Quantum: simulate Grover on the N-dim state vector in numpy (diffusion = 2|s⟩⟨s|−I, oracle = diag(±1)); count oracle applications with the BBHT unknown-M schedule; repeat 200 trials. Expect ∝ √(N/M) vs N/M; plot log-log slope ≈ 0.5 vs 1. ~1 h of work; a 1024-dim state vector is trivial.

**E2. Nearest-slot and top-k (Dürr–Høyer / k-maxima).** Same setup; implement Dürr–Høyer with rising threshold (each stage uses E1's routine), count total oracle calls until the true argmax is returned; then the k-maxima variant for k=16. Compare to N. Verifies O(√N) and O(√(kN)) constants and failure rate; measure the probability that the returned top-16 set equals numpy's `topk`.

**E3. Oracle cost with a finite-precision similarity.** Replace the exact oracle by a Hadamard-test estimate of ⟨q|key_i⟩ (7-qubit amplitude encoding, Qiskit statevector, or a numpy binomial model of shots). For ε ∈ {0.1, 0.05, 0.02}, count (Grover calls) × (shots per call) and record mis-marking near τ. This quantifies caveat (d)2 and shows the regime where the √ gain is eaten by precision cost; vary κ-driven margin γ.

## (f) Verdict

Promising only in the narrow, honest sense: for the *index-selection* sub-step of `lookup` and `write` (find a same-label slot above threshold; find top-k), there is a provable, tight Θ(√(N/M)) vs Θ(N/M) oracle-query gap that can be demonstrated exactly in numpy at N ≤ 1024. It is not promising as a route to improving `real_logit`/`slot_posterior` (dense sums, no search structure), and the gain is conditional on QRAM-style access whose per-step reload cost is Θ(ND) and must be reported as an assumption. The quantum associative-memory literature (Ventura, Trugenberger, Hopfield variants) does not transfer to real-valued cosine keys.

Single best experiment: **E1 extended with E2** — Grover/BBHT threshold search with unknown M over N ∈ {64,…,1024} random unit keys, oracle-call counts vs classical random probing and linear scan, then Dürr–Høyer top-k for k=16, reporting mean, 95th-percentile and success rate. It isolates exactly the claim that survives all caveats, needs only numpy, and produces a log-log slope plot that either shows 0.5 vs 1 or does not.
