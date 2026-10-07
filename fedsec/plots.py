"""Matplotlib plots for simulation results."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_accuracy(histories: dict[str, dict], out_path: str) -> None:
    """Line plot of test accuracy vs. round for each named run."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        ax.plot(hist["accuracy"], marker="o", markersize=3, label=name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Federated learning under attack: accuracy per round")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_asr_bars(histories: dict[str, dict], out_path: str) -> None:
    """Bar chart of final attack success rate per run."""
    names = list(histories)
    asrs = [histories[n]["final_asr"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(names)), asrs, color="tab:red", alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Attack success rate")
    ax.set_title("Backdoor attack success rate (final round)")
    ax.set_ylim(0, 1.02)
    for bar, v in zip(bars, asrs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.2f}",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
