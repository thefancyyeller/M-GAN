"""Experiment 5: the honest classical bar, random-hyperplane LSH on the memory.

Quantum top-k is compared against an *exact* linear scan in experiments
1-4.  Classically one would use approximate nearest-neighbour search; for
cosine similarity, random-hyperplane LSH (Charikar 2002) with L tables of b
bits gives sublinear query cost at the price of recall.  We measure, on the
trained memory with MNIST queries:

  distance evaluations per query  = size of the candidate set (union of the
                                    L buckets the query falls in)
  recall@16                       = fraction of the exact top-16 inside the
                                    candidate set's top-16
  write cost                      = L hash evaluations per written key

Sweeps b and L.  Writes exp5_lsh.csv.

    python research/exp5_lsh_baseline.py [--queries 64]
"""

import argparse
import csv

import numpy as np
import torch

from common import CHECKPOINT, out_path
from mgan import MemoryGAN, MemoryGANConfig
from mgan.data import images_only, mnist_loader

K = 16


def main(n_queries: int) -> None:
    rng = np.random.default_rng(5)
    config = MemoryGANConfig()
    model = MemoryGAN(config)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    keys = model.memory.keys.numpy()                                  # [N, D]
    loader = mnist_loader(config.data_root, n_queries, train=False)
    with torch.no_grad():
        queries = model.discriminator.encoder(images_only(next(iter(loader)))).numpy()
    n, d = keys.shape
    exact_top = np.argsort(-(queries @ keys.T), axis=1)[:, :K]

    out = out_path("exp5_lsh.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bits", "tables", "mean_candidates", "recall_at_1", "recall_at_16", "write_hashes", "N"])
        for bits in (6, 8, 10, 12):
            for tables in (1, 4, 16, 32, 64):
                planes = rng.standard_normal((tables, bits, d))
                key_codes = (np.einsum("tbd,nd->tnb", planes, keys) > 0)        # [T, N, b]
                key_codes = key_codes.dot(1 << np.arange(bits))                 # [T, N] ints
                q_codes = (np.einsum("tbd,qd->tqb", planes, queries) > 0).dot(1 << np.arange(bits))
                candidates_sizes, recall1, recall16 = [], [], []
                for qi in range(queries.shape[0]):
                    cand = np.unique(np.concatenate(
                        [np.flatnonzero(key_codes[t] == q_codes[t, qi]) for t in range(tables)]))
                    candidates_sizes.append(cand.size)
                    if cand.size == 0:
                        recall1.append(0.0)
                        recall16.append(0.0)
                        continue
                    sims = keys[cand] @ queries[qi]
                    got = cand[np.argsort(-sims)[:K]]
                    recall1.append(float(exact_top[qi, 0] in got))
                    recall16.append(np.isin(exact_top[qi], got).mean())
                w.writerow([bits, tables, f"{np.mean(candidates_sizes):.1f}", f"{np.mean(recall1):.3f}",
                            f"{np.mean(recall16):.3f}", tables, n])
                print(f"b={bits:2d} L={tables:2d}  candidates {np.mean(candidates_sizes):7.1f}"
                      f"  recall@1 {np.mean(recall1):.3f}  recall@16 {np.mean(recall16):.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=64)
    main(parser.parse_args().queries)
