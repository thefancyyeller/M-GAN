# Research plan: can quantum search improve the memoryGAN memory?

Question (owner's framing): can quantum algorithms improve memoryGAN in some
capacity, measured as query complexity / asymptotics, proven in isolated
sub-problems rather than end-to-end training?

Inputs: the four literature reviews in this folder (01 Grover fetch, 02
Dürr–Høyer k-NN, 03 amplitude estimation, 04 tooling and prior work) and
the measured statistics of the trained memory
(`research/characterize_memory.py`, `D:\M-GAN\outputs\research\memory_stats.json`).

## What the memory actually does, as search problems

| memory op | search form | classical cost | quantum candidate |
|---|---|---|---|
| `write`: find a same-label slot with sim above threshold | threshold search, M marked of N | N/M expected (random probe), N worst | BBHT: Θ(√(N/M)) |
| `lookup`: top-k by similarity | k largest of N | N (scan) | k-maxima: Θ(√(kN)) |
| `_oldest_slot`: argmax age | max of N | O(1) with an LRU list | not a target (straw man, report 04) |
| `real_logit`: softmax-weighted real fraction | amplitude of a prepared state | N (exact) or 1/ε² (sampling) | QAE: O(1/ε) uses of the prep unitary |
| `slot_posterior`: full N-vector | vector readout | N | no shortcut (report 03) |

Measured instance (N=4096, κ=10, MNIST queries after 500 steps): top-1
similarity 0.86, 16th 0.22, median 0.00; ~2 slots per query exceed 0.5; the
softmax puts 39% of mass on the top slot and 60% on the top 16, the rest
spread over thousands of near-zero slots; p(real|q) ≈ 0.46 ± 0.11.

## Hypotheses

H1 (threshold fetch). BBHT finds a slot above a similarity threshold in
c·√(N/M) oracle queries with c ≈ 2, against N/M classically, on real memory
rows as well as synthetic ones. At the real memory's M ≈ 2 this is ~70 vs
~2000 queries.

H2 (top-k and argmax under a budget). Dürr–Høyer and k-maxima reach 90%
exact recovery with a budget of c·√N and c·√(kN) for a small c (≪ the
textbook 22.5), and that budget drops below N at a crossover N we can
locate. Classical partial scan with the same budget B succeeds with
probability ≈ B/N (argmax) and far less for top-k.

H3 (posterior by amplitude estimation). Given the state-preparation unitary,
maximum-likelihood QAE's error in p(real|q) falls as 1/uses, classical
sampling as 1/√uses, on real posteriors. The state-preparation multiplier
√(1/p_eff) on the trained memory is of order √N, so the end-to-end bound is
O(√N/ε), not N-independent.

H4 (greedy counting, the owner's idea). Binary search on a similarity
threshold with quantum counting, then collection, recovers the top-k in
O(√N·polylog) queries: legitimately sublinear, but not better than k-maxima.

H5 (the classical bar). Random-hyperplane LSH on the same memory reaches
high recall@16 with far fewer than N distance evaluations, so the quantum
gain is over the *exact, preprocessing-free* scan only.

## Accounting (from report 04 §f)

One query = one application of the threshold/comparison oracle on one
classical index or on the whole superposition. Classical count = indices
evaluated. QAE charges uses of the preparation unitary A, the same unit as
one classical draw from the softmax. State preparation / QRAM is reported
separately and every quantum result is labelled "conditional on QRAM".

## Experiments (all exact amplitude-level numpy, `qmem/`)

| id | script | measures | hypotheses |
|---|---|---|---|
| E1 | `research/exp1_threshold_search.py` | mean/p95 queries vs N, M; BBHT vs random probe vs scan | H1 |
| E2 | `research/exp2_argmax_topk.py` | P(exact) vs budget c·√N / c·√(kN); quantum vs partial scan | H2 |
| E3 | `research/exp3_posterior_qae.py` | median abs error vs uses of A, QAE vs sampling; p_eff multiplier vs N | H3 |
| E4 | `research/exp4_greedy_counting.py` | queries and exact rate, greedy counting vs k-maxima | H4 |
| E5 | `research/exp5_lsh_baseline.py` | recall@16 vs distance evaluations, LSH on the real memory | H5 |
| E6 | `research/exp6_qiskit_check.py` | N=16 gate-level Grover (Qiskit Aer) vs the numpy model | validity of E1–E4 |

Sources for E1, E2, E4, E5: synthetic (uniform values / random unit keys)
and real (similarity rows of MNIST test images against the trained memory,
slots subsampled to N). Results and figures go to
`D:\M-GAN\outputs\research\`; the write-up is `REPORT.md`.
