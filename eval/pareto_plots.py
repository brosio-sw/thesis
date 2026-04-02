"""
Pareto trade-off plots: toxicity vs. fluency.

Each method is a point in the (mean_toxicity, mean_ppl) plane.
The Pareto front (lower toxicity AND lower perplexity) is highlighted.

Usage
-----
    python eval/pareto_plots.py --csv data/processed/results_table.csv \
                                --output_dir data/processed/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask of the Pareto-efficient points.

    A point is Pareto efficient if no other point is strictly better in
    *all* dimensions (lower is better for both toxicity and PPL).

    Args:
        costs: [N, 2] array where column-0 = toxicity, column-1 = PPL.

    Returns:
        Boolean array [N], True = Pareto efficient.
    """
    is_eff = np.ones(len(costs), dtype=bool)
    for i, c in enumerate(costs):
        if is_eff[i]:
            # dominate if other is ≤ in all and < in at least one
            is_eff[is_eff] = ~np.all(costs[is_eff] <= c, axis=1) | np.all(
                costs[is_eff] == c, axis=1
            )
            is_eff[i] = True
    return is_eff


def plot_pareto(
    df: pd.DataFrame,
    x_col: str = "mean_toxicity",
    y_col: str = "mean_ppl",
    label_col: str = "method",
    output_dir: str | Path = "data/processed/plots",
    filename: str = "pareto.pdf",
    figsize: tuple = (7, 5),
) -> None:
    """
    Scatter plot with Pareto front highlighted.

    Args:
        df:         DataFrame with at least x_col, y_col, and label_col.
        x_col:      Column name for x-axis (toxicity).
        y_col:      Column name for y-axis (fluency / PPL).
        label_col:  Column used as point labels.
        output_dir: Where to save the figure.
        filename:   Output filename.
        figsize:    Figure size.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    costs = df[[x_col, y_col]].values
    pareto_mask = is_pareto_efficient(costs)

    fig, ax = plt.subplots(figsize=figsize)

    # All points
    ax.scatter(
        df[x_col][~pareto_mask],
        df[y_col][~pareto_mask],
        c="steelblue", alpha=0.7, zorder=2, label="Methods",
    )
    # Pareto set
    pareto_df = df[pareto_mask].sort_values(x_col)
    ax.scatter(
        pareto_df[x_col],
        pareto_df[y_col],
        c="crimson", zorder=3, s=80, label="Pareto front",
    )
    ax.plot(
        pareto_df[x_col],
        pareto_df[y_col],
        c="crimson", lw=1.2, ls="--", zorder=2,
    )

    # Labels
    for _, row in df.iterrows():
        ax.annotate(
            str(row[label_col]),
            xy=(row[x_col], row[y_col]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean Toxicity Score ↓")
    ax.set_ylabel("Perplexity (GPT-2) ↓")
    ax.set_title("Toxicity–Fluency Trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=150)
    print(f"Pareto plot saved → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/results_table.csv")
    parser.add_argument("--output_dir", default="data/processed/plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    plot_pareto(df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
