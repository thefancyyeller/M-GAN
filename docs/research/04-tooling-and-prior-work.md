# 04 – Prior work, fine print, and simulation tooling

Scope: the three candidate quantum replacements for `mgan/memory.py` group-A
operations (N = 4096 unit-norm keys, D = 128, cosine similarity, top-k = 16,
argmax-age replacement, softmax-weighted real/fake posterior). Metrics are
oracle query counts and asymptotics, not wall-clock. Versions checked on
PyPI on 2026-11-17.

## (a) Summary

Nobody has published a quantum (or quantum-inspired) *key/value lookup*
inside a memoryGAN. The nearest things are (1) GANs whose "associative
memory" is a quantum sampler (QAAN, QCA-MolGAN): a *prior*, not a lookup;
(2) quantum k-NN / k-minima papers that give the O(√(kN)) top-k primitive
but apply it to static classification; (3) Grover-based sparse attention
(Gao–Song–Yang–Zhang), the closest algorithmic ancestor. New here is the
combination: Grover / Dürr–Høyer / QAE applied to the three concrete
operations of a memory rewritten every step, measured in oracle queries
against both brute force and sublinear classical ANN, with an explicit
state-preparation charge. All three ideas are safe from Tang-style
dequantization (quadratic, query-model speedups; BBBV proves Ω(√N) tight),
but all are exposed to Aaronson's fine print: the oracle must be built from
classical keys, which without QRAM costs O(N·D) per query. Argmax-age is
*not* a credible quantum target (an LRU list is O(1) classically).
Tooling: numpy amplitude simulator for everything at N ≤ 4096, Qiskit
2.6.2 + Aer 0.18.4 for the N = 16 gate-level cross-check, hand-rolled
IQAE/MLAE with the unmaintained `qiskit-algorithms` 0.4.0 only as a check.

## (b) Prior work

| Work | What it does | What is different from ours |
|---|---|---|
| Zoufal, Lucchi, Woerner, qGAN, npj QI 2019 ([arXiv:1904.00043](https://arxiv.org/abs/1904.00043)) | Variational quantum generator learns/loads a distribution; QAE used downstream for option pricing | Generator is quantum, no memory; QAE estimates a payoff, not a memory posterior |
| Stein et al., QuGAN, IEEE QCE 2021 ([arXiv:2010.09036](https://arxiv.org/abs/2010.09036)) | G and D both quantum; swap-test fidelity loss | Fully variational, no external memory, no query-complexity claim |
| Wilson et al., QAAN ([arXiv:1905.13205](https://arxiv.org/abs/1905.13205)) | Quantum Boltzmann machine as the "associative memory" of an associative adversarial network | Memory is a *sampler* of a latent distribution, not key/value slots with top-k lookup |
| Thomas et al., QCA-MolGAN, 2025 ([arXiv:2509.05051](https://arxiv.org/abs/2509.05051)) | QCBM prior matched to a discriminator layer, called associative memory | Same: no lookup, no write rule, no query model |
| Niu et al., EQ-GAN, PRL 2022 ([arXiv:2105.00080](https://arxiv.org/abs/2105.00080)) | Quantum GAN that learns a shallow circuit approximating a QRAM state | Addresses *loading* data, the very bottleneck our ideas depend on; could be a state-prep subroutine, not a memory |
| Rebentrost et al., Quantum Hopfield network, PRA 2018 ([arXiv:1710.03599](https://arxiv.org/abs/1710.03599)); Ventura–Martinez QuAM ([quant-ph/9807053](https://arxiv.org/abs/quant-ph/9807053)); Trugenberger ([quant-ph/0012100](https://arxiv.org/abs/quant-ph/0012100)) | Content-addressable memories: amplitude-encoded Hopfield via matrix inversion; Grover-style pattern completion | Associative *recall* of stored patterns; no real/fake values, no explicit top-k, no age-based overwrite; Hopfield version carries HHL-type state-prep caveats |
| Wiebe, Kapoor, Svore 2014 ([arXiv:1401.2142](https://arxiv.org/abs/1401.2142)); Basheer, Afham, Goyal k-NN ([arXiv:2003.09187](https://arxiv.org/abs/2003.09187)); Berti et al. QMI 2024 ([link](https://iris.unitn.it/retrieve/handle/11572/408151/759578/s42484-024-00155-2.pdf)); Gao, Ji, Wang k-minima ([arXiv:2412.16586](https://arxiv.org/abs/2412.16586)) | Quantum NN / k-NN via coherent distance + Dürr–Høyer; O(√(kM)) top-k; near-optimal approximate k-minima | Exactly our top-k primitive, but on static datasets for classification, compared against Monte Carlo or brute force; none handles a memory mutated every step or a GAN posterior |
| Gao, Song, Yang, Zhang, Grover attention ([arXiv:2307.08045](https://arxiv.org/abs/2307.08045)) | Grover selects the k large entries of an n×n attention matrix in Õ(n^1.5 k^0.5 d) | Closest ancestor: Grover top-k over dot products. Attention over a sequence, not a persistent key/value memory; theory only |
| Quixer ([arXiv:2406.04305](https://arxiv.org/abs/2406.04305)); QASA ([arXiv:2504.05336](https://arxiv.org/abs/2504.05336)); overlap-interference attention ([arXiv:2602.06699](https://arxiv.org/abs/2602.06699)) | Variational / LCU-QSVT quantum transformers, 2024–26 | No query-complexity claims, no memory writes |

Novelty statement: known oracle-model primitives applied to a key/value
memory that (i) is rewritten every step, (ii) has a label-conditioned
fetch, and (iii) feeds a softmax posterior whose precision is set by the
GAN loss, with accounting that charges state preparation. No prior work
does (i)–(iii).

## (c) Fine-print and dequantization risk per idea

Background. Aaronson, *Read the fine print*, Nat. Phys. 2015
([doi](https://www.nature.com/articles/nphys3272),
[author PDF](https://scottaaronson.com/papers/qml.pdf)): a claim must account
for input loading, data uniformity, and readout. Tang 2018
([arXiv:1807.04271](https://arxiv.org/abs/1807.04271)) and the
Chia–Gilyén–Li–Lin–Tang–Wang framework (STOC 2020) show *exponential*
speedups built on QRAM access to *low-rank* matrices collapse once the
classical side gets matching ℓ²-sample-and-query access. Unstructured
search is outside this: BBBV
([quant-ph/9701001](https://arxiv.org/abs/quant-ph/9701001)) proves Θ(√N)
optimal quantumly and Ω(N) classically. The catch moves to the oracle:
Jaques & Rattew, *QRAM: a survey and critique*, Quantum 2025
([arXiv:2305.10310](https://arxiv.org/abs/2305.10310)) argue any QRAM costs
Ω(N) hardware and Grover-with-QRAM does not beat a comparable classical
machine per query.

| Idea | Quantum queries | Classical bar | Dequantizable? | Fine print |
|---|---|---|---|---|
| **Grover fetch** (find a slot with label = ℓ and sim > τ, or the same-label slot among the top-k) | O(√(N/M)), M marked ([quant-ph/9605043](https://arxiv.org/abs/quant-ph/9605043)) | O(N/M) expected scan; but among the top-k it is O(k)=16 classically, so Grover only matters if applied to all N | No (BBBV) | Oracle must compute sign(q·kᵢ − τ) coherently: needs keys in QRAM or an O(N·D)-gate state-prep per query. Unknown M → needs BBHT exponential search, ~×2–3 overhead |
| **Dürr–Høyer top-k** (replace `lookup`) | O(√N) for min ([quant-ph/9607014](https://arxiv.org/abs/quant-ph/9607014)); O(√(kN)) for k smallest ([quant-ph/0401091](https://arxiv.org/abs/quant-ph/0401091)); N=4096,k=16 → ≈ 256·c vs 4096 | Brute force O(N); LSH O(N^ρ) with ρ<1 after preprocessing; HNSW empirically ~O(log N) | No in the query model; **but** the fair bar is sublinear ANN, which already beats O(√N) at this N | Same oracle cost; constants (DH's bound is 22.5√N, practical ≈ 3–5√N) must be measured not quoted; keys change every write, so any "preprocessing" must be charged per write on both sides |
| **QAE posterior** (estimate p(real\|q) = Σ_real softmax(κ sᵢ)) | O(1/ε) Grover iterations for additive ε ([quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055)); quadratic over Monte Carlo (Montanaro, [arXiv:1504.06987](https://arxiv.org/abs/1504.06987)) | Exact O(N) scan (what `real_logit` does); Monte Carlo O(1/ε²) *only if* you can sample slots cheaply | No for the MC-vs-QAE gap; **yes in spirit**: if the classical side gets the same sample access the exact O(N) scan at N=4096 is already cheaper than O(1/ε) QAE rounds unless ε ≫ 1/√N | Each Grover iterate needs A and A† where A prepares √softmax amplitudes over N slots: that is a full amplitude-encoding of the key set (O(N) gates, or QRAM). κ=10 gives e^{±10} dynamic range, so the "uniformity" caveat bites. Output is a scalar with no gradient to q: encoder needs a surrogate (already noted in `memory.py`) |

Argmax-age (`_oldest_slot`): do **not** present DH max-finding here. Ages
increment uniformly, so "oldest" = least-recently-touched: an LRU list is
O(1), a heap O(log N). √N against O(1) is a straw man.

Verdict: all three survive dequantization *as query-complexity results*;
none survives the fine print *end-to-end* without assuming QRAM. Report
every result twice (query model with QRAM; gate model with explicit state
prep) and say that only the first can show an advantage.

## (d) Classical baselines

| Baseline | Query cost | Notes |
|---|---|---|
| Linear scan (current `lookup`) | N similarity evaluations + O(N log k) for top-k | N = 4096, exact; insert O(1) |
| Monte Carlo posterior | O(1/ε²) samples, each one query | Needs cheap slot sampling; variance ∝ e^{2κ} range |
| LSH (Indyk–Motwani; Andoni–Indyk) | O(n^ρ) distance evaluations, ρ = log(1/p₁)/log(1/p₂) <1; ρ ≤ 1/c² for ℓ², ρ→1/(2c²−1) data-dependent ([arXiv:1306.1547](https://arxiv.org/abs/1306.1547)); cross-polytope LSH for cosine ([arXiv:1509.02897](https://arxiv.org/abs/1509.02897)) | For c = √2, ρ ≈ 0.5: **same exponent as Grover** at zero quantum cost; insert O(L) hashes; approximate, not exact |
| HNSW (Malkov–Yashunin, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)); FAISS ([arXiv:2401.08281](https://arxiv.org/abs/2401.08281)) | Empirically ~O(log N) distance evaluations; no worst-case bound; Indyk–Xu show linear-time adversarial instances ([arXiv:2310.19126](https://arxiv.org/abs/2310.19126)) | Insert ~O(log N); graph is maintained under our merge/replace writes |

So the honest top-k bar is not 4096 but "≈ 64–100 distance evaluations
with LSH/HNSW after preprocessing", versus DH's ≈ 3–5·√(kN) ≈ 800–1300
oracle calls. The quantum result is only interesting stated as *exact,
preprocessing-free, worst-case* √(kN).

## (e) Tooling (Windows 11, Python 3.13, numpy 2.4, torch 2.6 CPU)

| Tool (version, date) | Py 3.13 / Win | Install | Max qubits (16 GB, statevector) | Oracle-call counting | Grover / DH / QAE primitives |
|---|---|---|---|---|---|
| **numpy only** | yes | 0 MB | N-vector: any N ≤ 2²⁰; 2-D rotation model: unlimited | trivial (you call the oracle) | write yourself (~150 lines total) |
| **Qiskit 2.6.2** (2026-11-05; `numpy>=2,<3`, Py ≥3.10) ([PyPI](https://pypi.org/project/qiskit/)) + **qiskit-aer 0.18.4** (2026-11-17; cp313 win_amd64 wheel, 9.9 MB; `qiskit>=1.1,<3`, Py ≥3.11) ([PyPI](https://pypi.org/project/qiskit-aer/)) | yes | ~60 MB + Aer 10 MB | `Statevector` ≈ 25–26; Aer ≈ 28–29 | wrap oracle as a named `Gate`; `circuit.count_ops()["oracle"]` | `grover_operator`, `phase_estimation` in circuit library ([docs](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit_library)); AE algorithms live only in **qiskit-algorithms 0.4.0** (2025-08-29, Py 3.13 added, `qiskit>=1.0` unbounded, "no longer officially supported by IBM") ([PyPI](https://pypi.org/project/qiskit-algorithms/), [release notes](https://qiskit-community.github.io/qiskit-algorithms/release_notes.html)): `AmplitudeEstimation`, `IterativeAmplitudeEstimation`, `MaximumLikelihoodAmplitudeEstimation`, `FasterAmplitudeEstimation`, `Grover`, `EstimationProblem` |
| **PennyLane 0.45.1** (2026-06-26; Py ≥3.11, `numpy>=2.0`, `autoray==0.8.4` hard pin, pulls `pennylane-lightning>=0.45`) ([PyPI](https://pypi.org/project/pennylane/)) | yes (classifiers 3.11–3.14) | ~150 MB with lightning | `default.qubit` ≈ 25; `lightning.qubit` ≈ 28 | `qml.Tracker` / `qml.specs`; count by op name | `qml.GroverOperator`, `qml.AmplitudeAmplification`, `qml.QuantumPhaseEstimation`; no IQAE/MLAE; torch interface works with torch 2.6 (`interface="torch"`) |
| **Cirq 1.7.0** (2026-06-29; Py ≥3.11, `numpy~=2.1`, `matplotlib~=3.9`, `pandas`, `scipy~=1.15`) ([PyPI](https://pypi.org/project/cirq/)) | yes | ~120 MB (heavy pins) | ≈ 25 | manual | none (examples only) |
| **Qulacs 0.6.14** (2026-07-29; cp313 win_amd64 wheel) ([PyPI](https://pypi.org/project/qulacs/)) | yes | ~10 MB | ≈ 28–29 (fast C++) | manual | none |

Recommendation.

(i) **Exact amplitude simulations, N ≤ 4096: numpy only.** Grover and
QAE live in span{|good⟩,|bad⟩}: one iteration is a 2×2 rotation and
expected counts are exact functions of θ = asin√(M/N). Dürr–Høyer is
randomised (threshold index, measurement outcomes): simulate the
N-amplitude vector exactly (O(N) per oracle call), sample only the
measurements, report mean ± CI over ≥10⁴ seeded trials; for N ≤ 64 also
compute the expectation exactly as a Markov chain over thresholds.

(ii) **Gate-level validation, N = 16: Qiskit 2.6.2 + Aer 0.18.4.** 4 index
qubits + similarity register + ancilla ≈ 10–12 qubits; oracle as a labelled
`Gate`, assembled with `grover_operator`; run `Statevector` (exact) and Aer
`shots=10⁴`; assert per-iteration success probabilities match the numpy
model to 1e-10 and `count_ops()["oracle"]` equals the numpy counter.

(iii) **QAE: hand-roll canonical QAE, IQAE and MLAE in numpy.** All need
only the Bernoulli probability sin²((2m+1)θ) for Grover power m (canonical
outcome distribution: Brassard et al. Thm 11; IQAE:
[arXiv:1912.05559](https://arxiv.org/abs/1912.05559); MLAE:
[arXiv:1904.10246](https://arxiv.org/abs/1904.10246)). Keep
`qiskit-algorithms==0.4.0` in a *separate* venv only to cross-check IQAE's
interval schedule; it is unmaintained and may break on a later Qiskit 2.x.

```
py -3.13 -m venv D:\M-GAN\venv-q
D:\M-GAN\venv-q\Scripts\activate
python -m pip install --upgrade pip
pip install "numpy==2.4.*" "scipy>=1.14"
pip install "qiskit==2.6.2" "qiskit-aer==0.18.4"
:: optional cross-check only (unmaintained):
pip install "qiskit-algorithms==0.4.0"
:: optional second simulator opinion, only if wanted:
pip install "pennylane==0.45.1"
```

Windows gotchas: use the `py -3.13` launcher; keep this venv separate from
the M-GAN torch env (PennyLane's `autoray==0.8.4` and Cirq's
`matplotlib~=3.9`/`pandas` pins fight existing envs); Aer is an MSVC build
needing the VC++ 2015–2022 redistributable; set `OMP_NUM_THREADS` so Aer
does not take every core; if `pennylane-lightning` fails to import on 3.13
fall back to `default.qubit`; enable long paths for deep venv dirs.
Nothing was installed for this report.

## (f) Proposed oracle-query accounting convention

1. **The oracle.** O_q : |i⟩|b⟩ → |i⟩|b ⊕ f_q(i)⟩ with f_q(i) ∈ {0,1}
   (threshold or comparison bit) or O_q : |i⟩|0⟩ → |i⟩|s_i⟩ with
   s_i = ⌊2^r·q·kᵢ⌉ for DH/top-k. One query = one application of O_q or
   O_q†, on any input (classical basis state or superposition). The D-flop
   dot product inside the oracle costs the same on both sides and is not
   counted.
2. **Classical count** = number of distinct i for which f_q(i) was
   evaluated (scan: N; top-k scan: N; MC: number of samples; LSH/HNSW:
   candidate-set size + probe distance evaluations). Preprocessing and
   per-write maintenance (hash inserts, graph updates) are reported in a
   separate "write cost" column, amortised per `write` call.
3. **Quantum count** = Σ oracle applications. Grover: one per iteration.
   DH: one per Grover iteration *plus* one per threshold comparison (the
   comparator is an oracle call). QAE: S_χ counts as one oracle query per
   Grover iterate; A and A† are counted separately as **state-prep calls**
   (n_prep), never folded into n_oracle. Reflections about |ψ⟩ or |0⟩ are
   free (data-independent).
4. **Expected totals.** Report E[queries to success] = Σ over restarts,
   i.e. queries/p_success, computed from exact amplitudes, not from a single
   lucky run; for DH report mean, median, 95th percentile.
5. **State prep / QRAM charge.** Two columns, always both:
   (Q) *QRAM model*: uniform superposition free, O_q = 1 query, QRAM
   contents updated at O(1) per written slot (charge in the write column);
   (G) *gate model*: no QRAM; each O_q or A costs N·D elementary gates,
   reported as gate count. A result that only wins in column (Q) is
   labelled "conditional on QRAM" in every table.
6. **Precision.** QAE targets are specified as additive ε on p(real|q)
   with failure probability α; classical MC gets the same (ε, α) via
   Hoeffding; the exact scan has ε = 0.
7. **Sizes.** N ∈ {16, 64, 256, 1024, 4096}, k = 16, M ∈ {1, 16, 256};
   fit log(queries) vs log N and report the exponent with CI, next to the
   theoretical 0.5 / 1.0.
