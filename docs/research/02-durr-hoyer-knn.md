# 02 — Dürr–Høyer minimum finding and quantum k-NN for memory recall

Scope: the two argmax/argmin operations in `mgan/memory.py` — (a) `lookup`: top-k slots by cosine similarity (k=16, N=4096); (b) `write`: `_oldest_slot`, argmax over integer ages. Question: do quantum minimum-finding / k-NN algorithms give a provable query-complexity improvement, demonstrable in a small simulation?

## (a) Summary

Dürr–Høyer (DH) finds an argmin of an N-entry table with O(√N) oracle queries, and Dürr–Heiligman–Høyer–Mhalla (DHHM) find the k smallest with Θ(√(kN)) queries, both with success probability ≥ 1/2 per run and a matching Ω lower bound. Both map directly onto our two operations; the speedup is *provable in the oracle-query model* and *exactly simulable classically* (Grover dynamics live in a 2-D subspace), so a small-scale demonstration is cheap. Caveats: (i) the ages argmax has a trivial exact oracle but is also trivially O(log N) classically with a heap, so the quantum version there is a testbed, not an improvement; (ii) the similarity top-k needs the cosine similarity computed *coherently* (swap test + amplitude estimation, cost 1/ε per query, plus QRAM-style key access) or pre-loaded; (iii) at N=4096 the constants (c ≈ a few, times 1/ε) erase a √N-vs-N gap of only 64×; (iv) classical approximate NN (HNSW/LSH) is already sublinear heuristically. Verdict: promising as a rigorous, isolated query-count result; not a practical win at this memory size.

## (b) Key papers

- Dürr & Høyer 1996, *A quantum algorithm for finding the minimum* — https://arxiv.org/abs/quant-ph/9607014 . Threshold loop over BBHT exponential search; success ≥ 1/2 within 22.5√N + 1.4 lg²N steps.
- Boyer, Brassard, Høyer, Tapp 1998, *Tight bounds on quantum searching* — https://arxiv.org/abs/quant-ph/9605034 . Exponential search for unknown t marked items, λ = 6/5 (any 1 < λ < 4/3 works), expected ≤ (9/2)·m₀ ≈ (9/4)√(N/t) iterations; lower bound sin(π/8)√N.
- Zalka 1999, *Grover's quantum searching algorithm is optimal* — https://arxiv.org/abs/quant-ph/9711070 . (π/4)√N is asymptotically tight for search; BBBV 1997 (DH's cited lower bound) gives Ω(√N) — https://arxiv.org/abs/quant-ph/9701001 .
- Dürr, Heiligman, Høyer, Mhalla 2006, *Quantum query complexity of some graph problems*, SICOMP 35(6) — https://arxiv.org/abs/quant-ph/0401091 . Theorem 6: k smallest values in O(√(kN)) bounded-error queries (Remark 1: c√(kN) queries, success ≥ 1/2; extend the run rather than restart). Theorem 19: Ω(√(kN)) via Ambainis's adversary method.
- Miyamoto, Iwamura, Kise 2019, *A quantum algorithm for finding k-minima* — https://arxiv.org/abs/1907.03315 . Simpler O(√(kN)) variant (threshold near the k-th value + amplitude amplification).
- van Apeldoorn, Gilyén, Gribling, de Wolf 2020, *Quantum SDP-solvers*, Quantum 4, 230 — https://arxiv.org/abs/1705.01843 . "Generalized minimum finding": minimum over values prepared *in superposition by a unitary* (no direct value oracle), cost O(1/‖ψ_k*‖) = O(√N) for uniform weights.
- Gao, Ji, Wang 2024 / ESA 2025, *Quantum approximate k-minimum finding* — https://arxiv.org/abs/2412.16586 . Õ(√(kN)) queries to an (ε,δ)-approximate value oracle; with amplitude estimation to precision ε, Õ(√(kN)/ε). This is the correct model for a swap-test similarity oracle.
- Quek, Canonne, Rebentrost 2020, *Robust quantum minimum finding* — https://arxiv.org/abs/2003.11777 . Noisy comparisons; Õ(√(N(1+Δ))), Δ = number of near-ties.
- Wiebe, Kapoor, Svore 2014, *Quantum algorithms for nearest-neighbor methods* — https://arxiv.org/abs/1401.2142 . Inner-product NN: O(√M ⌈d² r_max⁴/ε⌉ log(M/δ)) queries (explicit constant 1080·…) via coherent swap test + amplitude estimation + DH; classical Monte Carlo O(M d² r_max⁴/ε²).
- Basheer, Afham, Goyal 2020 — https://arxiv.org/abs/2003.09187 . Fidelity k-NN reduces to quantum k-maxima, O(√(kM)).
- Li, Lin, Yu, Guo 2021 — https://arxiv.org/abs/2103.04253 . Hamming-distance k-NN with DH; the quadratic speedup is lost in high dimension.
- Quantum multi-label k-NN 2026 — https://arxiv.org/abs/2607.21919 . c-SWAP + amplitude estimation + k-maximal search, O(√(km)); same pipeline, no new bound.
- GB-QkNN, IJCAI 2025 — https://arxiv.org/abs/2505.23066 . "Quantum HNSW", claims O(log M); no Grover/AE, per-layer O(M) scan unaccounted — not a query-model result; treat as unproven.
- Gilliam, Woerner, Gonciulea 2021, *Grover adaptive search* — https://arxiv.org/abs/1912.04088 . DH loop as used in practice (λ a free parameter > 1; implementations commonly use 8/7 or 6/5). Wang, Xu, Zhang 2026 — https://arxiv.org/abs/2601.13195 — do k-minimum finding without QRAM.

## (c) Complexity statements

| task | classical | quantum (oracle queries) |
|---|---|---|
| argmax of N values | N−1 comparisons, Θ(N); O(log N)/update with a heap | DH: expected ≤ (45/4)√N + (7/10)lg²N; P(success) ≥ 1/2 after 22.5√N + 1.4lg²N (Markov on 2× expectation); c runs → 1−2^(−c); lower bound Ω(√N), search-tight constant π/4 |
| k smallest of N | O(N) selection (quickselect) or O(N log k) heap, after O(ND) to compute sims | DHHM: c√(kN), P ≥ 1/2; Θ(√(kN)) tight. Approximate-value oracle: Õ(√(kN)) (Gao–Ji–Wang) |
| k smallest by cosine sim, sims not precomputed | O(ND) exact; HNSW/LSH ≈ O(log N)·D heuristic | Õ(√(kN)/ε) preparations of \|key_j⟩ (swap test + AE to precision ε); each preparation is a QRAM read or O(D) gates |

Rule of thumb: the quantum count beats N when C√(kN)/ε < N, i.e. N > C²k/ε². With C≈3, k=16, ε≈0.01 (gap between the 16th and 17th sim) this is N ≳ 10⁶; for the ages case (ε irrelevant, exact oracle) C√N < N as soon as N > ~10, but the heap wins anyway.

## (d) Assumptions and caveats

1. **Oracle model.** DH/DHHM count queries to T[j]; for ages, T is a classical integer compare — trivially coherent. For similarities, T[j] = ⟨q, key_j⟩ must be produced *inside* the oracle: (i) precompute sims classically (O(ND)) and load into a register — then quantum saves nothing end-to-end; (ii) swap test between |q⟩ and |key_j⟩ with amplitude estimation (WKS; Gao–Ji–Wang's (ε,δ)-oracle), costing O(1/ε) key preparations per query and needing QRAM or a key-loading circuit. Only (ii) is a genuine speedup, and only in the query model with QRAM assumed.
2. **Success probability.** All results are ≥ 1/2 per budget; boost by extending the run (DHHM Remark 1) or repeating. The measured output of BBHT is uniform over marked items — this is what makes DH's rank analysis (p(t,r)=1/r) work and must be preserved in a simulation.
3. **Exact vs approximate top-k.** With ε-approximate sims the returned set is a (k, O(ε))-approximate minimum set; near-ties cost extra (Quek et al.'s Δ). The memory tolerates this: `write` uses only indices+labels, and `lookup`'s gradient-carrying sims can be recomputed classically for the k returned indices at O(kD). `real_logit` and `slot_posterior` are dense over all N and are untouched by any top-k speedup.
4. **Classical competition.** Brute-force at N=4096, D=128 is 0.5 MFLOP; HNSW/LSH give sublinear heuristic recall. The quantum claim is a *worst-case, exact-recall, query-model* statement — a different, defensible claim; say so explicitly.
5. **Differentiability.** A sampled swap test is not differentiable; recompute sims classically for the returned indices, as the `memory.py` contract requires.

## (e) Experiments (numpy; amplitude-level, exact)

Simulation primitive: with t marked out of N, after r Grover iterations P(hit) = sin²((2r+1)θ), sin θ = √(t/N); on hit the sample is uniform over marked, else uniform over unmarked. One oracle query per Grover iteration, plus one to verify the sample classically.

```
BBHT(marked, N, lam=6/5):            # marked: boolean array, t unknown to the algorithm
    m = 1; queries = 0
    loop:
        r = randint(0, ceil(m)-1)      # uniform in {0,...,ceil(m)-1}
        queries += r + 1               # r iterations + 1 verification
        t = marked.sum(); p = sin((2r+1)*asin(sqrt(t/N)))**2
        j = random marked index if rand() < p else random unmarked index
        if marked[j]: return j, queries
        m = min(lam*m, sqrt(N))        # BBHT: lam in (1, 4/3); 6/5 (paper) or 8/7 (common)

DH_argmin(T, N):                       # record first-hitting time
    y = randint(N); queries = 0; hit = None
    while hit is None:
        j, q = BBHT(T < T[y], N); queries += q
        if T[j] < T[y]: y = j
        if T[y] == T.min(): hit = queries
    return hit                         # true minimum first held after `hit` queries

DHHM_kmin(T, N, k):                    # Basheer/Miyamoto-style simplification of DHHM Thm 6
    S = random k distinct indices; thr = max(T[S]); queries = 0
    until set(S) == true top-k:        # record first-hitting time
        j, q = BBHT(T < thr, N); queries += q
        if T[j] < thr: S[argmax(T[S])] = j; thr = max(T[S])
```
(Replace "T <" by "sims >" for similarity; use strict inequality with index tie-break so t is well defined. When t = 0 BBHT never hits — in the real algorithm the time budget 22.5√N + 1.4 lg²N is the only stop; the simulation knows the answer, so it records the first-hitting time instead and the budget giving P ≥ 1/2 is its median.)

**E1 — DH argmax over ages.** N ∈ {16, 32, …, 1024}, 2000 trials; ages drawn (a) distinct, (b) with ties as in a warmed-up memory. Report the median first-hitting query count (= budget giving P ≥ 1/2) and mean; fit c·√N; compare with N−1 and with the 22.5√N bound. Sweep λ ∈ {6/5, 8/7, 1.31}. Expect c ≈ 2–5 ≪ 22.5.

**E2 — DHHM top-k by similarity (precomputed sims as oracle).** Keys: 1024 random unit vectors in R¹²⁸ (or dump `memory.keys` from the smoke test), queries from the same distribution. k ∈ {1, 2, 4, 8, 16}, N ∈ {64, …, 1024}. Report median queries until S equals the exact top-k; fit c·√(kN); overlay N−1 (classical selection reads each sim once). Also report near-tie sensitivity: fraction of runs where the k-th/(k+1)-th gap < 0.01.

**E3 — Qiskit oracle validation (N = 8–16, k = 1).** Build the DH oracle as a fixed-point comparator over a sims register loaded by a QROM; run BBHT with the schedule above on the statevector simulator; check the hit probabilities and the uniform-over-marked output against E1's closed form. Optional add-on: a swap-test + 3-bit amplitude-estimation oracle for D = 4 keys to exhibit the 1/ε cost in gate count. This validates the oracle-model assumption that E1/E2 rely on; it is not where the complexity claim is proven.

## (f) Verdict

Promising *as a rigorous, isolated demonstration*: DH/DHHM give a provable Θ(√(kN)) vs Θ(N) query separation that applies verbatim to `_oldest_slot` and to `lookup` (indices only), and the amplitude-level simulation is exact, so E1/E2 yield query-count curves at N ≤ 1024 in seconds. Not promising as a practical replacement: at N = 4096 the √ gap is 64× for argmax and 16× for k = 16, the similarity oracle costs 1/ε per query plus QRAM, and the ages argmax is O(log N) classically with a heap. **Single best experiment: E2** — median oracle queries to recover the exact top-k set versus N and k, with fitted constant c in c√(kN), plotted against the N−1 classical scan and the 22.5√N DH bound; it measures precisely the owner's question, and its one real assumption (an exact sims oracle) is stated, not hidden.
