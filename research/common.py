"""Shared helpers for the research scripts: real similarity rows and output paths."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgan import MemoryGAN, MemoryGANConfig          # noqa: E402
from mgan.data import images_only, mnist_loader      # noqa: E402

CHECKPOINT = Path(r"D:\M-GAN\outputs\memorygan_500.pt")
OUT_DIR = MemoryGANConfig().output_root / "research"


def out_path(name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / name


def real_similarity_rows(n_queries: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarities of MNIST test queries to the trained memory.

    Returns ``(sims [Q, N], is_real [N])`` as numpy arrays.  This is the
    actual problem instance a quantum lookup inside the memoryGAN would face.
    """
    torch.manual_seed(seed)
    config = MemoryGANConfig()
    model = MemoryGAN(config)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    loader = mnist_loader(config.data_root, n_queries, train=False)
    with torch.no_grad():
        q = model.discriminator.encoder(images_only(next(iter(loader))))
        sims = model.memory.similarity(q).numpy()
    return sims, (model.memory.values == 1.0).numpy()


def subsample_slots(row: np.ndarray, n: int, rng: np.random.Generator,
                    keep: np.ndarray | None = None) -> np.ndarray:
    """Restrict one similarity row to ``n`` random slots (always keeping ``keep``)."""
    if n >= row.shape[0]:
        return row
    keep = np.array([], dtype=int) if keep is None else np.asarray(keep)
    others = np.setdiff1d(np.arange(row.shape[0]), keep)
    chosen = np.concatenate([keep, rng.choice(others, n - keep.size, replace=False)])
    return row[chosen]
