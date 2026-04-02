import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_probe_r2_scores(csv_path: str, output_path: str):
    """
    Reads the summary.csv containing probe metrics and plots the R2 score
    across different model layers.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Could not find {csv_file}")
        return

    # Load data
    df = pd.read_csv(csv_file)
    
    # Check if necessary columns exist
    if 'layer' not in df.columns or 'val_r2' not in df.columns:
        print("Error: The CSV does not contain 'layer' or 'val_r2' columns.")
        return
        
    df = df.sort_values(by="layer")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['layer'], df['val_r2'], marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    
    # Formatting
    plt.title('Probe R² Score Evolution Across Layers', fontsize=14, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Validation R² Score', fontsize=12)
    plt.xticks(df['layer'], rotation=90)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot R2 scores of linear probes across layers.")
    parser.add_argument(
        "--csv", 
        type=str, 
        default="data/probes_llada_masked/full_run/masked_only_probes/summary.csv", 
        help="Path to the summary.csv file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="probe_r2_scores.png", 
        help="Path to save the generated plot"
    )
    
    args = parser.parse_args()
    plot_probe_r2_scores(args.csv, args.output)
