#!/usr/bin/env python3
"""Plot Pure LoRA versus MemRift+LoRA 100-step training loss curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "full-weight-only-20260928T1745Z"
OUTPUT_DIR = Path(__file__).resolve().parent

MODELS = {
    "LLaMA-3.1-8B": "llama8b",
    "Aquila2-7B": "aquila",
}

METHODS = {
    "Pure LoRA": ("pure_lora_training_curve.csv", "#0072B2", "o"),
    "MemRift + LoRA": ("memrift_lora_training_curve.csv", "#D55E00", "s"),
}


def load_curve(model_key: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
    path = (
        REPO_ROOT
        / "output"
        / "gsm8k_100iter"
        / model_key
        / RUN_ID
        / "results"
        / filename
    )
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    steps = np.asarray([int(row["iteration"]) for row in rows])
    losses = np.asarray([float(row["loss"]) for row in rows])
    if len(steps) != 100 or not np.array_equal(steps, np.arange(1, 101)):
        raise ValueError(f"expected exactly steps 1..100 in {path}")
    return steps, losses


def centered_moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if window % 2 != 1:
        raise ValueError("moving-average window must be odd")
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )


def main() -> int:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.15), sharex=True, constrained_layout=True)

    for panel_index, (ax, (model_name, model_key)) in enumerate(zip(axes, MODELS.items())):
        for method, (filename, color, marker) in METHODS.items():
            steps, losses = load_curve(model_key, filename)
            smoothed = centered_moving_average(losses, window=7)

            # Preserve all observed values while using a smoothed line to make
            # the optimization trend legible.
            ax.plot(steps, losses, color=color, alpha=0.18, linewidth=0.8, zorder=1)
            ax.plot(
                steps,
                smoothed,
                label=method,
                color=color,
                linewidth=2.0,
                marker=marker,
                markevery=[9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )
            ax.scatter([100], [losses[-1]], color=color, marker=marker, s=28, zorder=4)
            ax.annotate(
                f"{losses[-1]:.3f}",
                xy=(100, losses[-1]),
                xytext=(-9, 17 if method == "MemRift + LoRA" else -20),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=7.5,
                color=color,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                zorder=5,
            )

        ax.set_title(f"({chr(97 + panel_index)}) {model_name}", pad=7)
        ax.set_xlabel("Training step")
        ax.set_xlim(1, 103)
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.set_axisbelow(True)
        if panel_index == 0:
            ax.set_ylabel("Language-model loss")
            ax.legend(loc="upper right")

    fig.suptitle("100-Step Fine-Tuning Loss: Pure LoRA vs. MemRift + LoRA", fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "Faint lines: per-step loss; solid lines: centered 7-step moving average; labels: step-100 loss.",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "fig_training_loss_comparison.png"
    pdf_path = OUTPUT_DIR / "fig_training_loss_comparison.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
