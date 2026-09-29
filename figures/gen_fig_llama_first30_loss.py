#!/usr/bin/env python3
"""Plot two separate first-30-step LLaMA loss figures for the LoRA runs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "full-weight-only-20260928T1745Z"
RESULT_DIR = REPO_ROOT / "output" / "gsm8k_100iter"
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_KEY = "llama8b"
NUM_STEPS = 30

PURE_COLOR = "#0072B2"
MEMRIFT_COLOR = "#D55E00"


def load_loss(variant: str) -> tuple[np.ndarray, np.ndarray]:
    path = (
        RESULT_DIR
        / MODEL_KEY
        / RUN_ID
        / "results"
        / f"{variant}_training_curve.csv"
    )
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[:NUM_STEPS]

    steps = np.asarray([int(row["iteration"]) for row in rows])
    loss = np.asarray([float(row["loss"]) for row in rows])
    if not np.array_equal(steps, np.arange(1, NUM_STEPS + 1)):
        raise ValueError(f"expected exactly steps 1..{NUM_STEPS} in {path}")
    return steps, loss


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.9,
        }
    )


def save_loss_figure(
    steps: np.ndarray,
    loss: np.ndarray,
    method: str,
    color: str,
    marker: str,
    output_name: str,
    y_limits: tuple[float, float],
) -> None:
    """Save one method in its own figure with shared comparison axes."""
    fig, ax = plt.subplots(figsize=(6.75, 3.65), constrained_layout=True)
    ax.plot(
        steps,
        loss,
        color=color,
        marker=marker,
        markevery=2,
        markersize=4.0,
        zorder=3,
    )
    ax.set_title(f"LLaMA-3.1-8B {method} Training Loss (First 30 Steps)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, NUM_STEPS)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(7))

    output_stem = OUTPUT_DIR / output_name
    for suffix in ("pdf", "svg"):
        output_path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(output_path)
        print(f"saved: {output_path}")
    plt.close(fig)


def main() -> int:
    configure_style()
    steps, pure = load_loss("pure_lora")
    mem_steps, memrift = load_loss("memrift_lora")
    if not np.array_equal(steps, mem_steps):
        raise ValueError("training-step mismatch between Pure LoRA and MemRift + LoRA")

    combined_min = float(min(pure.min(), memrift.min()))
    combined_max = float(max(pure.max(), memrift.max()))
    padding = 0.03 * (combined_max - combined_min)
    y_limits = (combined_min - padding, combined_max + padding)

    save_loss_figure(
        steps,
        pure,
        method="Pure LoRA",
        color=PURE_COLOR,
        marker="o",
        output_name="fig_llama_first30_loss_pure_lora",
        y_limits=y_limits,
    )
    save_loss_figure(
        steps,
        memrift,
        method="MemRift + LoRA",
        color=MEMRIFT_COLOR,
        marker="s",
        output_name="fig_llama_first30_loss_memrift_lora",
        y_limits=y_limits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
