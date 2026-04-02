import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def plot_lambda_sweep(json_path: str, output_path: str):
    """
    Read a lambda sweep summary JSON and plot:
    - sent_answer_mean
    - ppl_answer_mean
    - invalid_fraction

    Expected default layout:
    data/alignment_variants_v4/full_run/
        diffmean_holdout_lambda_sweep_fixed/
            real_full_pooled/
                lambda_sweep_summary.json
    """
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Error: Could not find {json_file}")
        return

    with open(json_file, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        print(f"Error: JSON at {json_file} is empty or not a dict.")
        return

    rows = []
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if "lambda" not in entry:
            continue
        rows.append(entry)

    if len(rows) == 0:
        print(f"Error: No valid lambda entries found in {json_file}")
        return

    rows.sort(key=lambda x: x.get("lambda", float("inf")))

    lambdas = [row.get("lambda") for row in rows]
    sent_answer_means = [row.get("sent_answer_mean", float("nan")) for row in rows]
    ppl_answer_means = [row.get("ppl_answer_mean", float("nan")) for row in rows]
    invalid_fractions = [row.get("invalid_fraction", float("nan")) for row in rows]

    variant = rows[0].get("variant", "unknown_variant")
    fill_strategy = rows[0].get("fill_strategy", "unknown_fill")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Sentiment Answer Mean vs Lambda
    axes[0].plot(lambdas, sent_answer_means, marker="o", linestyle="-", linewidth=2, markersize=6)
    axes[0].set_title("Sentiment Answer Mean vs Lambda", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Lambda", fontsize=11)
    axes[0].set_ylabel("Sentiment Mean (Negativity)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Perplexity Answer Mean vs Lambda
    axes[1].plot(lambdas, ppl_answer_means, marker="s", linestyle="-", linewidth=2, markersize=6)
    axes[1].set_title("Perplexity Answer Mean vs Lambda", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lambda", fontsize=11)
    axes[1].set_ylabel("Perplexity (PPL)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Invalid Fraction vs Lambda
    axes[2].plot(lambdas, invalid_fractions, marker="^", linestyle="-", linewidth=2, markersize=6)
    axes[2].set_title("Invalid Fraction vs Lambda", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Lambda", fontsize=11)
    axes[2].set_ylabel("Invalid Fraction", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(
        f"DiffMean Holdout Lambda Sweep | variant={variant} | fill={fill_strategy}",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot metrics vs lambda for the fixed DiffMean holdout sweep."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=(
            "data/alignment_variants_v4/full_run/"
            "diffmean_holdout_lambda_sweep_fixed/"
            "real_full_pooled/"
            "lambda_sweep_summary.json"
        ),
        help="Path to the lambda_sweep_summary.json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diffmean_holdout_lambda_sweep_fixed_real_full_pooled.png",
        help="Path to save the generated plot",
    )

    args = parser.parse_args()
    plot_lambda_sweep(args.json, args.output)