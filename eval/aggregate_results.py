"""
Aggregate per-run metric files (JSON/CSV) into comparison tables.

Usage
-----
    python eval/aggregate_results.py --results_dir data/generations \
                                     --output_dir data/processed

Each run should have a metadata JSON file of the form::

    {
      "method":      "H1+S1",
      "remasking":   "low_confidence",
      "steering":    "mean_steering",
      "alpha":       15.0,
      "mean_toxicity": 0.12,
      "toxic_fraction": 0.08,
      "mean_ppl":    42.3,
      ...
    }

The script reads all such files and outputs:
  * ``results_table.csv``  — one row per run.
  * ``results_table.md``   — Markdown table for easy copy-paste.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load all ``*_metrics.json`` files under `results_dir` into a DataFrame."""
    results_dir = Path(results_dir)
    rows = []
    for path in sorted(results_dir.rglob("*_metrics.json")):
        with open(path) as f:
            data = json.load(f)
        data["source_file"] = path.name
        rows.append(data)

    if not rows:
        raise FileNotFoundError(f"No *_metrics.json files found in {results_dir}")

    return pd.DataFrame(rows)


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned summary table sorted by mean_toxicity (ascending).
    Drops the raw `scores` list column if present.
    """
    drop_cols = [c for c in df.columns if c in {"scores", "per_text_ppl", "per_text_distinct_n"}]
    summary = df.drop(columns=drop_cols, errors="ignore")
    if "mean_toxicity" in summary.columns:
        summary = summary.sort_values("mean_toxicity")
    return summary.reset_index(drop=True)


def save_results(
    df: pd.DataFrame,
    output_dir: str | Path,
    basename: str = "results_table",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{basename}.csv"
    md_path  = output_dir / f"{basename}.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"Saved CSV  → {csv_path}")
    print(f"Saved MD   → {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate generation metrics.")
    parser.add_argument(
        "--results_dir", default="data/generations",
        help="Directory containing *_metrics.json files.",
    )
    parser.add_argument(
        "--output_dir", default="data/processed",
        help="Output directory for aggregated tables.",
    )
    args = parser.parse_args()

    df = load_results(args.results_dir)
    summary = make_summary_table(df)
    save_results(summary, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
