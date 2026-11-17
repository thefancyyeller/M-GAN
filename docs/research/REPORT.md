# Can quantum search improve the memoryGAN memory? Results

Branch `research/quantum-memory`, 2026-11-17. Question: can Grover search,
Dürr–Høyer minimum finding or amplitude estimation improve the memoryGAN
memory (`mgan/memory.py`) in query complexity, shown in isolated
sub-problems rather than end-to-end training. Everything below is an exact
amplitude-level simulation in numpy (`qmem/`), cross-checked against a
gate-level Qiskit circuit at N = 16. Figures and CSVs are in
`D:\M-GAN\outputs\research\`. Literature reviews: `01`–`04` in this folder;
hypotheses and accounting: `PLAN.md`.

## Bottom line

1. **Yes, for the memory's search steps, in the oracle-query model.** Finding a
   slot above a similarity threshold (what `write` does to find a same-label
   neighbour) costs about 1.6·√(N/M) oracle queries with BBHT against N/M
   classically, on real memory rows exactly as on synthetic ones. At the
   trained memory's typical M ≈ 2 marked slots of N = 4096 that is ~70
   queries versus ~1300–1900. Finding the nearest slot costs 6·√N queries
   for 97% success (384 at N = 4096) where a classical scan of 384 slots
   succeeds 11% of the time.
2. **Marginal for exact top-16 at the current memory size.** k-maxima needs
   ~12·√(16N) queries for 96%+ exact recovery: 3072 at N = 4096 versus 4096
   for the scan. The √(kN) scaling is real but the constant makes the
   crossover N ≈ 2300; it is a clear win only for a memory several times
   larger than today's.
3. **No, for the posterior.** Amplitude estimation does give the textbook
   1/ε versus 1/√ε error scaling on real posteriors, but it needs a
   state-preparation unitary whose coherent cost on the trained memory is
   ~39 similarity-oracle calls per use (the softmax is extremely peaked), so
   end-to-end it costs more than the exact N-term sum at N = 4096 for any
   useful ε.
4. **The greedy-counting idea works but is not competitive.** Binary search on
   a threshold with quantum counting, then collection, recovers the exact
   top-16 essentially always and scales as about N^0.4, but costs 1100–5600
   queries at N = 64–4096, always above N. It is dominated by k-maxima.
5. **The classical bar is not the linear scan.** On this memory,
   random-hyperplane LSH finds the nearest slot with 100% recall from ~20
   candidate distance evaluations, cheaper than any quantum variant. It
   does poorly at exact top-16 (recall 0.85 needs ~2600 candidates) because
   neighbours 3–16 sit at near-zero similarity. The quantum results are
   therefore interesting as *exact, worst-case, preprocessing-free* bounds,
   and as the only sublinear option for the top-16 set on this data.

Everything quantum is **conditional on QRAM-style access** to the keys
(report 04 §c): without it, building the oracle costs N·D gates per query and
no speedup survives. The memory is rewritten every step, so the keys must be
re-loaded per step; the per-query gains only pay off amortised over many
queries between writes.

## What the memory does, measured

From `research/characterize_memory.py` on the 500-step checkpoint, 256 MNIST
test queries, N = 4096, κ = 10:

| statistic | value |
|---|---|
| top-1 / top-16 / median similarity | 0.86 / 0.22 / 0.00 |
| top-1 minus top-2 gap, mean / min | 0.070 / 0.0015 |
| slots with similarity > 0.5 | ~2 of 4096 (M/N = 0.0005) |
| softmax mass on top-1 / top-16 / top-64 | 39% / 60% / 63% |
| p(real \| q), mean ± std | 0.46 ± 0.11 |

Two consequences. The search problems are in the sparse-marked regime where
Grover gains most. The posterior has 37% of its mass spread over thousands
of near-zero slots, which is why top-k approximations of it break training
(DESIGN.md) and why its coherent state preparation is expensive.

## Accounting

One query = one evaluation of the threshold predicate on one classical index,
or one phase-flip of the whole superposition. Classical counts = indices
evaluated. QAE counts uses of the preparation unitary A, the same unit as one
classical draw from the softmax. State preparation is reported separately.
`qmem/oracle.py` enforces this on both sides.

## Results

### E1 Threshold search ([`fig_exp1_threshold.png`](figures/fig_exp1_threshold.png), `exp1_threshold.csv`)

Mean oracle queries, 200 trials per cell, real memory rows:

| N | M | BBHT | random probe | scan | √(N/M) |
|---|---|---|---|---|---|
| 256 | 1 | 25.7 | 250 | 126 | 16 |
| 1024 | 1 | 54.6 | 956 | 497 | 32 |
| 4096 | 1 | 105 | 4030 | 1998 | 64 |
| 4096 | 2 | 70.6 | 1897 | 1328 | 45 |
| 4096 | 16 | 25.0 | 222 | 238 | 16 |

BBHT = (1.5–1.7)·√(N/M), success 100%; log-log slope 0.5 against 1.0. Synthetic
and real rows agree to within noise, so the result does not depend on the
similarity distribution. This is the `write` rule's same-label neighbour
fetch.

### E2 Nearest slot and exact top-16 under a budget ([`fig_exp2_budget.png`](figures/fig_exp2_budget.png), `exp2_budget.csv`)

P(correct) versus budget B, real rows, 100 trials:

| task | N | B for ≥ 90% quantum | classical P at that B | B = N classical |
|---|---|---|---|---|
| nearest slot | 256 | 96 (6√N) | 0.34 | 256 |
| nearest slot | 1024 | 192 (6√N) | 0.19 | 1024 |
| nearest slot | 4096 | 384 (6√N) | 0.11 | 4096 |
| exact top-16 | 256 | 768 (12√(kN)) | 1.00 (B > N) | 256 |
| exact top-16 | 1024 | 1536 (12√(kN)) | 1.00 (B > N) | 1024 |
| exact top-16 | 4096 | 3072 (12√(kN)) | 0.01 | 4096 |

The textbook Dürr–Høyer budget (22.5√N) is 4× looser than needed; c ≈ 6
suffices. For top-16 the measured constant is c ≈ 12 in √(kN) units, i.e.
48√N, so the method only beats N once N > 2300.

### E3 Posterior by amplitude estimation ([`fig_exp3_qae.png`](figures/fig_exp3_qae.png), `exp3_qae.csv`, `exp3_prep.csv`)

Median |estimate − p(real|q)|, 20 real queries × 20 repeats:

| uses of A | MLAE (quantum) | sampling (classical) |
|---|---|---|
| 128 | 0.020 | 0.029 |
| 1 120 | 0.0031 | 0.0091 |
| 16 608 | 0.00021 | 0.0026 |

Fitted slopes −0.93 (quantum) and −0.50 (classical): the 1/ε separation
holds. State-preparation multiplier √(1/p_eff), p_eff = Z/(N·w_max):

| source | N = 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| random keys | 2.1 | 2.9 | 3.4 | 3.9 |
| trained memory | 2.1 | 2.7 | 3.9 | **39** |

On the full trained memory one slot carries most of the softmax mass, so a
coherent preparation of A costs ~39 oracle calls per use. For ε = 0.01 QAE
needs ~300 uses → ~12 000 oracle calls, versus 4096 for the exact sum. There
is no end-to-end gain for the posterior at this size, and the estimate
carries no gradient to the encoder.

### E4 Greedy counting ([`fig_exp4_greedy.png`](figures/fig_exp4_greedy.png), `exp4_greedy.csv`)

| N | greedy counting | exact rate | k-maxima at 6√(kN) | exact rate |
|---|---|---|---|---|
| 64 | 1199 | 0.98 | 192 | 0.30 |
| 1024 | 3936 | 1.00 | 769 | 0.34 |
| 4096 | 5627 | 0.98 | 1538 | 0.16 |

Growth ≈ N^0.4, so sublinear, but each quantum-counting step costs ~560 uses
and ~10 steps are needed. A legitimate algorithm, not a better one.

### E5 Classical LSH on the trained memory ([`fig_exp5_lsh.png`](figures/fig_exp5_lsh.png), `exp5_lsh.csv`)

| bits | tables | candidates | recall@1 | recall@16 |
|---|---|---|---|---|
| 10 | 4 | 20 | 1.00 | 0.20 |
| 12 | 16 | 22 | 1.00 | 0.11 |
| 8 | 64 | 959 | 1.00 | 0.69 |
| 6 | 64 | 2567 | 1.00 | 0.86 |

Nearest-slot retrieval is a solved classical problem here (20 distance
evaluations plus L hashes per write). Exact top-16 is not: the 3rd–16th
neighbours are at similarity ≈ 0.2 and below, indistinguishable from the bulk
by hashing. That is the one place the exact √(kN) quantum bound has no
cheap classical competitor on this data.

### E6 Gate-level check (`exp6_qiskit.csv`)

Qiskit 2.6.2 + Aer, N = 16, M ∈ {1, 2, 4}, 0–4 Grover iterations: exact
statevector probabilities match sin²((2j+1)θ) to 1e-10, Aer shots and the
numpy model agree to sampling noise, and the oracle-gate count equals the
numpy query count. The amplitude model used in E1–E4 is faithful.

## What this means for M-GAN

Where a quantum memory could plausibly help, in order:

1. **The write rule's neighbour fetch.** Replace "look up the top-k, pick the
   nearest same-label slot" by a BBHT threshold search with the label folded
   into the oracle (mark slot i iff label matches and sim > τ). Provable
   Θ(√(N/M)) vs Θ(N/M); on this memory M is 1–2, the best case. The oracle
   output is an index, which is all `write` needs, so the differentiability
   contract is untouched.
2. **The generator's conditioning and `lookup` as index selection.**
   Dürr–Høyer for the nearest slot at 6√N; sims for the returned indices are
   recomputed classically (O(kD)) so gradients still flow. Worth it in the
   query model; classical LSH is cheaper in practice for the top-1.
3. **Exact top-k at larger N.** The √(kN) bound is the only sublinear exact
   method on this data, but with c ≈ 12 it needs N ≳ 2300 to pay, and a
   finite-precision similarity oracle will add a 1/ε factor (report 02).

Where it does not help: the posterior `real_logit` (dense sum, peaked softmax,
no gradient), the ages argmax (O(1) with an LRU list), and anything that
needs the full `slot_posterior` vector.

## Caveats carried forward

- QRAM / key loading is assumed and must be stated in every claim.
- All oracles here are exact. A swap/Hadamard-test similarity adds O(1/ε)
  per query and mis-marks near-threshold slots; the gap between the 16th and
  17th similarity on this memory is small, so exact top-16 will suffer most.
- Success probabilities are per budget; repetition multiplies cost by log(1/δ).
- Query counts are not time. The classical scan is 0.5 MFLOP at N = 4096.

## Suggested next experiments

1. **Quantum-fetch write rule inside the memory.** Subclass `Memory` with
   `write` using BBHT over a label-aware oracle, run the existing training
   loop with the oracle counter on, and report queries per step versus the
   classical rule and whether training dynamics change. Isolated, cheap.
2. **Finite-precision oracle.** Replace the exact comparator by a Hadamard-test
   estimate with ε ∈ {0.1, 0.05, 0.02}; measure total (Grover calls × shots)
   and mis-marking, to find where precision eats the √ gain.
3. **Scale N.** Rerun E2 at N = 16k–64k (numpy handles it) to show the top-16
   crossover and fit the exponent with confidence intervals.
4. **Hybrid.** LSH candidates for the top-1, Grover only for the exact top-k
   tail; this is where the two methods complement each other on this data.

## Reproduce

```
python research/characterize_memory.py
python research/exp1_threshold_search.py
python research/exp2_argmax_topk.py
python research/exp3_posterior_qae.py
python research/exp4_greedy_counting.py
python research/exp5_lsh_baseline.py
D:\M-GAN\venv-q\Scripts\python.exe research/exp6_qiskit_check.py
python research/plot_results.py
python -m pytest tests
```
