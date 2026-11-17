"""Draw the research figures from the experiment CSVs.

    python research/plot_results.py

Reads D:\\M-GAN\\outputs\\research\\exp*.csv and writes PNGs beside them.
Colours follow a fixed categorical order (blue, orange, aqua, yellow) so a
method keeps its colour across figures; lines are thin, grids recessive.
"""

import csv
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "legend.frameon": False, "lines.linewidth": 2,
})
SERIES = {"quantum": "#2a78d6", "classical": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK = "#0b0b0b"


def read(name: str) -> list[dict]:
    with (OUT_DIR / name).open() as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {path}")


def fig_exp1() -> None:
    rows = read("exp1_threshold.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, source in zip(axes, ("synthetic", "real")):
        for m, style in ((1, "-"), (16, "--")):
            for method, colour, label in (("bbht", SERIES["quantum"], "BBHT (quantum)"),
                                          ("random_probe", SERIES["classical"], "random probe (classical)")):
                pts = sorted((int(r["N"]), float(r["mean_queries"])) for r in rows
                             if r["source"] == source and int(r["M"]) == m and r["method"] == method)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style, color=colour,
                        label=f"{label}, M={m}")
        ns = np.array([64, 256, 1024, 4096])
        ax.plot(ns, ns, ":", color="#898781", linewidth=1, label="N")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("slots N")
        ax.set_title(f"{source} values", color=INK, loc="left")
    axes[0].set_ylabel("mean oracle queries to find a marked slot")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("E1  Threshold search: queries vs N (M marked slots)", color=INK, x=0.01, ha="left")
    save(fig, "fig_exp1_threshold.png")


def fig_exp2() -> None:
    rows = read("exp2_budget.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, ("argmax", "top16")):
        for n, alpha in ((256, 0.45), (4096, 1.0)):
            pts = sorted((float(r["c"]), float(r["quantum_success"]), float(r["classical_success"]))
                         for r in rows if r["source"] == "real" and r["task"] == task and int(r["N"]) == n)
            cs = [p[0] for p in pts]
            ax.plot(cs, [p[1] for p in pts], "-", color=SERIES["quantum"], alpha=alpha,
                    label=f"quantum, N={n}")
            ax.plot(cs, [p[2] for p in pts], "-", color=SERIES["classical"], alpha=alpha,
                    label=f"partial scan, N={n}")
        ax.axhline(0.9, color="#898781", linewidth=1, linestyle=":")
        unit = "sqrt(N)" if task == "argmax" else "sqrt(16 N)"
        ax.set_xlabel(f"query budget, in units of {unit}")
        ax.set_title(("nearest slot" if task == "argmax" else "exact top-16"), color=INK, loc="left")
        ax.set_xscale("log", base=2)
    axes[0].set_ylabel("P(correct), real memory rows")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("E2  Success vs budget: Durr-Hoyer / k-maxima against a partial classical scan",
                 color=INK, x=0.01, ha="left")
    save(fig, "fig_exp2_budget.png")


def fig_exp3() -> None:
    rows = read("exp3_qae.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, source in zip(axes, ("random", "real")):
        pts = sorted((int(r["uses_of_A"]), float(r["qae_median_abs_err"]), float(r["classical_median_abs_err"]))
                     for r in rows if r["source"] == source)
        uses = np.array([p[0] for p in pts])
        ax.plot(uses, [p[1] for p in pts], "-", color=SERIES["quantum"], label="MLAE (quantum)")
        ax.plot(uses, [p[2] for p in pts], "-", color=SERIES["classical"], label="sampling (classical)")
        ax.plot(uses, 1.0 / uses, ":", color=SERIES["quantum"], linewidth=1, label="1 / uses")
        ax.plot(uses, 0.5 / np.sqrt(uses), ":", color=SERIES["classical"], linewidth=1, label="0.5 / sqrt(uses)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("uses of the state-preparation unitary A")
        ax.set_title(f"{source} posteriors", color=INK, loc="left")
    axes[0].set_ylabel("median |estimate - p(real|q)|")
    axes[1].legend(fontsize=8)
    fig.suptitle("E3  Amplitude estimation of p(real|q): error vs budget", color=INK, x=0.01, ha="left")
    save(fig, "fig_exp3_qae.png")


def fig_exp4() -> None:
    rows = read("exp4_greedy.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, colour, label in (("greedy_counting", SERIES["aqua"], "greedy counting (owner's idea)"),
                                  ("k_maxima", SERIES["quantum"], "k-maxima, budget 6 sqrt(kN)")):
        pts = sorted((int(r["N"]), float(r["mean_queries"])) for r in rows
                     if r["source"] == "real" and r["method"] == method)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=colour, label=label)
    ns = np.array([64, 256, 1024, 4096])
    ax.plot(ns, ns, ":", color=SERIES["classical"], linewidth=1, label="N (classical scan)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("slots N")
    ax.set_ylabel("mean oracle queries (real memory rows)")
    ax.legend(fontsize=8)
    ax.set_title("E4  Top-16 retrieval: greedy counting vs k-maxima vs N", color=INK, loc="left")
    save(fig, "fig_exp4_greedy.png")


def fig_exp5() -> None:
    rows = read("exp5_lsh.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, colour, label in (("recall_at_1", SERIES["classical"], "recall@1"),
                               ("recall_at_16", SERIES["yellow"], "recall@16")):
        pts = sorted((float(r["mean_candidates"]), float(r[key])) for r in rows)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=22, color=colour, label=label,
                   edgecolors="#fcfcfb", linewidths=0.8)
    ax.axvline(4096, color="#898781", linewidth=1, linestyle=":")
    ax.text(4096, 0.05, " N = 4096", color="#898781", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("distance evaluations per query (LSH candidate set)")
    ax.set_ylabel("recall on the trained memory")
    ax.legend(fontsize=8, loc="center left")
    ax.set_title("E5  Classical LSH bar: recall vs candidates", color=INK, loc="left")
    save(fig, "fig_exp5_lsh.png")


if __name__ == "__main__":
    for fn in (fig_exp1, fig_exp2, fig_exp3, fig_exp4, fig_exp5):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skipped {fn.__name__}: {e}")
