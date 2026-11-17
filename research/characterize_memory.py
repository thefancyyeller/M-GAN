"""Measure what a quantum lookup would actually face in the trained memory.

Loads the classical memory from a checkpoint, encodes MNIST queries, and
reports the statistics that fix the difficulty of each quantum sub-problem:
how many slots pass a similarity threshold (Grover's M/N), the top-1/top-2
gap (how distinguishable the argmax is), how concentrated the softmax
weights are (whether top-k approximates the dense posterior), and the
real-slot fraction (what amplitude estimation must estimate).

    python research/characterize_memory.py [--checkpoint PATH] [--queries 256]

Writes JSON to ``config.output_root / research / memory_stats.json``.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgan import MemoryGAN, MemoryGANConfig
from mgan.data import images_only, mnist_loader


def main(checkpoint: Path, n_queries: int) -> None:
    config = MemoryGANConfig()
    model = MemoryGAN(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    memory = model.memory

    loader = mnist_loader(config.data_root, n_queries, train=False)
    with torch.no_grad():
        q = model.discriminator.encoder(images_only(next(iter(loader))))
        sims = memory.similarity(q)                                   # [Q, N]
        sorted_sims, _ = sims.sort(dim=1, descending=True)
        weights = torch.softmax(config.kappa * sims, dim=1)
        sorted_w, _ = weights.sort(dim=1, descending=True)
        is_real = memory.values == 1.0

    stats = {
        "slots": memory.n_slots,
        "dim": memory.dim,
        "kappa": config.kappa,
        "queries": int(q.shape[0]),
        "real_fraction": float(is_real.float().mean()),
        "top1_sim_mean": float(sorted_sims[:, 0].mean()),
        "top16_sim_mean": float(sorted_sims[:, 15].mean()),
        "median_sim_mean": float(sorted_sims[:, memory.n_slots // 2].mean()),
        "top1_top2_gap_mean": float((sorted_sims[:, 0] - sorted_sims[:, 1]).mean()),
        "top1_top2_gap_min": float((sorted_sims[:, 0] - sorted_sims[:, 1]).min()),
        "weight_on_top1_mean": float(sorted_w[:, 0].mean()),
        "weight_on_top16_mean": float(sorted_w[:, :16].sum(1).mean()),
        "weight_on_top64_mean": float(sorted_w[:, :64].sum(1).mean()),
        "p_real_mean": float(torch.sigmoid(memory.real_logit(q)).mean()),
        "p_real_std": float(torch.sigmoid(memory.real_logit(q)).std()),
        # Marked-slot counts for Grover-style threshold search: M(tau)/N.
        "marked_fraction_by_tau": {
            str(tau): float((sims > tau).float().mean()) for tau in (0.5, 0.7, 0.8, 0.9, 0.95)
        },
        # Among the k nearest, how many are real (what top-k counting would see).
        "real_among_top16_mean": float(is_real[sims.topk(16, dim=1).indices].float().mean()),
    }

    out_dir = config.output_root / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "memory_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:\M-GAN\outputs\memorygan_500.pt"))
    parser.add_argument("--queries", type=int, default=256)
    args = parser.parse_args()
    main(args.checkpoint, args.queries)
