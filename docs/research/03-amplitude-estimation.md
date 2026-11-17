# 03 - Quantum amplitude estimation for the memory posterior

Scope: estimating `real_posterior(q)` / `slot_posterior(q)` (mgan/memory.py) and cosine
similarities with quantum amplitude estimation (QAE) and quantum counting. Query
complexity only; no latency claims.

## Summary

QAE estimates an amplitude `a = sin^2(theta)` to additive error `eps` with `O(1/eps)`
uses of the state-preparation unitary `A`, against `O(1/eps^2)` classical samples, and
`Omega(1/eps)` is a proven lower bound, so nothing beats it. Our discriminator value
`p(real|q) = sum_i w_i v_i / sum_i w_i`, `w_i = exp(kappa s_i)`, is exactly such an
amplitude *if* a unitary `A` prepares `sum_i sqrt(w_i/Z) |i>|v_i>`. The catch is that
`A` depends on `q` through all `N` similarities: preparing it from classical weights costs
`O(N D)` (no gain), and preparing it coherently (QRAM key load + Hadamard-test inner
product + softmax-weight rotation + amplitude amplification) costs
`O(sqrt(N w_max / Z))` per use. The honest end-to-end bound is therefore
`O(sqrt(N w_max/Z) / eps) <= O(sqrt(N)/eps)` oracle queries: a quadratic gain in `N` and
in `eps` over classical exact `O(N D)` and over classical uniform-sample importance sampling
`O((N w_max/Z)/eps^2)`, but *not* N-independent. The QFT-free variants (MLAE, IQAE,
Aaronson-Rall) make the `1/eps` curve measurable in a numpy statevector with `N <= 1024`.
The gradient problem remains: a QAE estimate is a scalar, so the encoder needs a surrogate
gradient. Verdict: the `1/eps` scaling is real, cheap to demonstrate, and publishable as a
small isolated result; the `N` scaling is a Grover-type `sqrt(N)` at best and hinges on a
coherent similarity oracle the project does not have.

## Key papers

- Brassard, Hoyer, Mosca, Tapp 2002, *Quantum amplitude amplification and estimation*,
  https://arxiv.org/abs/quant-ph/0005055. Canonical QAE: with `M` applications of `Q`,
  `|a~ - a| <= 2 pi sqrt(a(1-a))/M + pi^2/M^2` w.p. >= 8/pi^2; counting `t` of `N` marked
  to relative error `eps` in `O(sqrt(N/t)/eps)` queries.
- Nayak, Wu 1999, https://arxiv.org/abs/quant-ph/9804066. `Omega(1/eps)` lower bound for
  mean/amplitude estimation; QAE is optimal in `eps`.
- Suzuki et al. 2020 (MLAE), https://arxiv.org/abs/1904.10246. No QFT, no controlled-Q;
  maximum likelihood over shots from circuits with `m_k` Grover iterations; Heisenberg
  scaling up to logs. Noisy-hardware follow-up: https://arxiv.org/abs/2006.16223.
- Grinko, Gacon, Zoufal, Woerner 2021 (IQAE), https://arxiv.org/abs/1912.05559. Rigorous
  confidence interval; about `(50/eps) log(2/alpha log2(pi/4eps))` oracle calls.
- Aaronson, Rall 2020, https://arxiv.org/abs/1908.10846. Grover-only approximate counting
  with a clean proof: `K` of `N` to multiplicative `eps` in `O(sqrt(N/K)/eps)` queries.
- Nakaji 2020 (FAE), https://arxiv.org/abs/2003.02417. Smaller constants than MLAE/IQAE.
- Rall, Fuller 2023, https://arxiv.org/abs/2207.08628. QSP framing; unbiased, non-destructive
  QAE and depth-for-shots trade-off. Low-depth variants: https://arxiv.org/abs/2012.03348,
  https://arxiv.org/abs/2410.01173.
- Montanaro 2015, https://arxiv.org/abs/1504.06987. Mean of a bounded/variance-bounded
  variable in `O(1/eps)` uses of the sampling circuit (our weighted fraction is this).
- Cornelissen, Hamoudi, Jerbi 2022, https://arxiv.org/abs/2111.09787. Multivariate mean
  estimation: estimating the full `slot_posterior` vector (dimension `N`) gains only when
  queries `>= N`; no shortcut for the whole vector.
- Lloyd, Mohseni, Rebentrost 2013, https://arxiv.org/abs/1307.0411; Wiebe, Kapoor, Svore
  2015, https://arxiv.org/abs/1401.2142. Coherent distance/inner-product estimation via
  swap test + QAE, nearest neighbour via Durr-Hoyer minimum finding
  (https://arxiv.org/abs/quant-ph/9607014); all assume QRAM state access.
- Kerenidis, Prakash 2016, https://arxiv.org/abs/1603.08675. The binary-tree QRAM data
  structure: `O(polylog)` state preparation per row, `O(polylog)` classical update per
  entry write (compatible with our one-row-per-write memory).
- van Apeldoorn, Gribling, Nieuwboer 2024, https://arxiv.org/abs/2302.10244. Find all `k`
  marked in optimal `O(sqrt(Nk))` queries; relevant to top-k retrieval.
- Guo, Yu, Agrawal, Rebentrost 2024, https://arxiv.org/abs/2402.16714. Coherent row-wise
  softmax over inner products via QSVT (degree `O(kappa + log 1/eps)`): the state-of-the-art
  recipe for the `exp(kappa s_i)` weight step.
- Zoufal, Lucchi, Woerner 2019 (qGAN), https://arxiv.org/abs/1904.00043. GAN used to *load*
  a distribution that QAE then integrates; opposite direction to ours, but shows QAE and GAN
  training compose. Lloyd-Weedbrook https://arxiv.org/abs/1804.09139 and Dallaire-Demers,
  Killoran https://arxiv.org/abs/1804.08641: fully quantum GANs, no memory.
- GAN noise tolerance: Zhou et al. ICLR 2019, *Don't let your discriminator be fooled*
  (robustness needed only in expectation over samples); *Robust GANs against dishonest
  adversaries*, https://arxiv.org/abs/1802.09700 (the vanilla GAN is *not* robust to a
  perturbed discriminator; Lipschitz-type regularisation restores it). Together these
  support tolerating unbiased `eps` noise in `D` under a regularised objective.

## Complexity of estimating p(real|q) to additive eps

Let `Z = sum_i w_i`, `a = p(real|q)`, `w_max = max_i w_i`, `p_eff = Z/(N w_max)` (the
fraction of slots carrying softmax mass; `~1` for a flat posterior, `~1/N` for a peaked one).

| Method | Cost | Assumes |
|---|---|---|
| Classical exact (memory.py) | `O(N D)` flops | nothing |
| Classical MC, sample `i ~ w/Z` | `O(a(1-a)/eps^2)` samples | sampler needs `Z`, i.e. `O(N D)` first |
| Classical MC, uniform `i`, self-normalised importance weights | `O((1/p_eff)/eps^2)` samples x `O(D)` | no precomputation |
| QAE with given `A` | `O(1/eps)` uses of `A` (BHMT, Montanaro); optimal (Nayak-Wu) | `A` prepares `sum sqrt(w_i/Z)|i>|v_i>` |
| `A` from classical weights (Grover-Rudolph / Mottonen) | `O(N)` gates + `O(N D)` classical | no gain |
| `A` coherent (QRAM load, Hadamard test, QSVT softmax, amplify) | `O(sqrt(1/p_eff))` inner-product oracle calls per use | QRAM over keys, `kappa` bounded |
| **End-to-end quantum** | `O(sqrt(N w_max/Z)/eps) <= O(sqrt(N)/eps)` | QRAM + coherent oracle |
| Quantum counting of real slots with `s_i > tau` (unweighted, `t` marked) | `O(sqrt(N/t)/eps_rel)` (BHMT, Aaronson-Rall) | threshold comparator at resolution `delta` costs `O(1/delta)` extra per query |

Similarity estimation alone: the swap test yields `s^2` (sign lost); the Hadamard test
yields `(1+s)/2`, so QAE gives `s` to `eps` in `O(1/eps)` state preparations versus
`O(1/eps^2)` shots. At `D = 128` the classical dot product is 128 flops, so single-similarity
QAE is never useful; it only pays inside a coherent sum over `i`.

## Assumptions and caveats

1. **State prep dominates.** The `1/eps` gain is unconditional given `A`; the `N` gain
   needs QRAM-style coherent access to keys (`|i>|0> -> |i>|k_i>`) with `O(log)`-depth
   updates. Without it the quantum method is strictly worse than `O(N D)`.
2. **Peaked softmax hurts.** With `kappa = 10` and a trained memory, `p_eff ~ 1/N`, so the
   amplification factor is `~sqrt(N)`: the Grover regime. Random keys in `R^128` give
   `p_eff ~ 0.07` (`kappa s ~ N(0, 0.78)`), so the early-training regime is cheaper.
3. **Logit error amplification.** The GAN consumes `log(a/(1-a))`; error `eps` in `a` becomes
   `eps/(a(1-a))` in the logit, worst exactly when `D` is confident.
4. **No gradient.** QAE returns a scalar. The encoder trains through `d p / d q`; it would
   need a parameter-shift or finite-difference pass (`D = 128` extra estimates) or a frozen
   encoder / surrogate gradient, as the memory.py contract already warns.
5. **slot_posterior as a vector** gets no shortcut (Cornelissen et al.); only scalar
   functionals of it (the real fraction, a single slot's probability) benefit.
6. Noisy hardware saturates the `1/eps` curve (Tanaka et al.); irrelevant for statevector
   simulation but relevant to any later hardware claim.

## The "greedy" idea, read charitably

(a) *Threshold search for top-k*: binary-search `tau` so that `#{i : s_i > tau} ~ k`, with
quantum counting at each step. Cost `O(sqrt(N/k) log(1/delta_tau) / eps_rel)` queries versus
classical `O(N)`: a legitimate `sqrt(N)` result, but the known optimal algorithm
(find-all-`k`-marked / `k`-minima, `O(sqrt(Nk))`) already does this without the search.
(b) *Greedy amplification to sample from `slot_posterior`*: amplitude amplification on the
coherently prepared softmax state samples a slot in `O(sqrt(1/p_eff))` oracle calls; this
is what the state-prep row of the table already charges. Neither idea improves the
asymptotics beyond the end-to-end bound above; (a) is the cleaner experiment because the
oracle is binary and needs no softmax step.

## Experiments (numpy, laptop CPU, N <= 1024)

Common machinery, no circuit library. Build `psi` from a real memory snapshot and query:
`psi_i = sqrt(w_i/Z)` on `|i>`; the "good" subspace is `{i : v_i = 1}`, `a = sum_good psi_i^2`,
`theta = asin(sqrt(a))`. Grover operator for a known prep `A` (`A|0> = psi`):
`Q = -A S_0 A^dag S_chi` with `S_0 = I - 2|0><0|`, `S_chi = I - 2 P_good`, so
`Q = -(I - 2 psi psi^T)(I - 2 P_good)`; apply as `v = x - 2 P_good x; Qx = -(v - 2 psi (psi . v))`,
`O(N)` per application. After `m` applications the good-state probability is
`sin^2((2m+1) theta)`; assert this against the explicit statevector. Query accounting:
one shot with `m` iterations costs `2m+1` uses of `A`/`A^dag` (report both this and
"oracle calls" `= m`).

- **MLAE**: schedule `m_k = 0, 1, 2, 4, ..., 2^(K-1)`, `N_shots` each; `h_k ~ Binom(N_shots,
  sin^2((2m_k+1)theta))`; maximise
  `L(theta) = sum_k h_k log sin^2((2m_k+1)theta) + (N_shots - h_k) log cos^2((2m_k+1)theta)`
  on a fine grid of `theta in [0, pi/2]`; `a~ = sin^2(theta~)`. Cramer-Rao error
  `~ 1/sqrt(sum_k N_shots (2m_k+1)^2)`.
- **IQAE**: keep `[theta_l, theta_u]`; pick the largest `k` with `(4k+2)[theta_l, theta_u]`
  inside one half-period of `sin^2`; shoot, Clopper-Pearson-bound the probability, invert to
  tighten the interval; stop at width `eps`.
- **Canonical QAE** (optional): sample outcome `y in {0..M-1}` with
  `Pr[y] = (1/2) sum_pm sin^2(M pi d_pm)/(M^2 sin^2(pi d_pm))`, `d_pm = y/M -/+ theta/pi`,
  `a~ = sin^2(pi y/M)`; cost `M - 1` oracle calls.
- **Classical baselines**: `n` draws from `w/Z` give `|mean - a| ~ sqrt(a(1-a)/n)`; uniform
  draws with self-normalised weights have variance inflated by `1/p_eff`.

**E1 (core).** For 20 random queries on a 1024-slot memory (random keys, and a snapshot after
a few hundred `write`s), run MLAE and IQAE with budgets `Q = 10 ... 10^5`, plot median
`|a~ - a|` vs `Q` on log-log, overlay MC `1/sqrt(n)` and the `pi/Q` line. Expected: slope
`-1` vs `-1/2`. Also plot logit error to show caveat 3.

**E2 (state-prep honesty).** From the same snapshots compute `p_eff = Z/(N w_max)` per query
and the implied multiplier `sqrt(1/p_eff)`; plot total cost `sqrt(1/p_eff)/eps` vs `N` for
`N in {64, ..., 1024}` against `N D` and `(1/p_eff)/eps^2`. This decides whether the
`sqrt(N)` claim holds on real memories at `kappa = 10`.

**E3 (greedy threshold / counting).** Marked set `{i : s_i > tau and v_i = 1}`; count with
Aaronson-Rall-style QAE at relative error 0.1 and binary-search `tau` to hit `k = 16`;
record total oracle calls vs `N` and compare with classical `N` and the `sqrt(Nk)` line.

## Verdict

Promising as an isolated, provable result: the `O(1/eps)` vs `O(1/eps^2)` separation for
`p(real|q)` is textbook, optimal, and demonstrable in a few hundred lines of numpy. Not
promising as an N-scaling story: the best honest bound is `O(sqrt(N)/eps)`, it depends on
a coherent similarity oracle plus QRAM, and it carries a `sqrt(N)` state-prep penalty
exactly in the trained, peaked-softmax regime. The gradient contract in memory.py is a
second hard blocker. Single best experiment: **E1 with E2 overlaid**, the measured `1/eps`
curve for MLAE/IQAE on real memory snapshots with the state-preparation multiplier on the
same axis, so the claim and its cost appear in one figure.
